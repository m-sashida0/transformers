from typing import Optional, Tuple

import math
import torch
import torch.nn.functional as F

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). 
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) 
    to (batch, num_attention_heads, seqlen, head_dim).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    module: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    # 1) スケーリング係数
    scale_factor = scale if scale is not None else 1.0 / math.sqrt(query.size(-1))

    # 2) バイアス行列の初期化 (dtype/device を query に合わせる)
    L, S = query.size(-2), key.size(-2)
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    # 3) causal mask の適用
    if is_causal:
        assert attn_mask is None, "When is_causal=True, attn_mask must be None"
        causal_mask = torch.tril(torch.ones(L, S, dtype=torch.bool, device=query.device))
        attn_bias = attn_bias.masked_fill(~causal_mask, float("-inf"))

    # 4) 外部 mask(attn_mask) の適用
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias = attn_bias.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_bias = attn_bias + attn_mask.to(attn_bias.dtype)

    # 5) GQA 用に key/value を繰り返す
    if enable_gqa:
        group_factor = query.size(-3) // key.size(-3)
        key = key.repeat_interleave(group_factor, dim=-3)
        value = value.repeat_interleave(group_factor, dim=-3)

    # 6) Attention 重みの計算
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    attn_weights = attn_weights + attn_bias
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_weights = F.dropout(attn_weights, dropout_p, training=(module.training if module is not None else True))

    # 7) Intervention (fix_layer/fix_head) の適用
    if module is not None and hasattr(module, "fix_layer") and hasattr(module, "layer_idx"):
        if module.fix_layer is not None and module.layer_idx is not None:
            bsz, n_heads, tgt_len, src_len = attn_weights.shape
            for f_layer, f_head in zip(module.fix_layer, module.fix_head):
                if module.layer_idx == f_layer:
                    if getattr(module, "fix_temperature", None) is not None:
                        idx = module.fix_layer.index(f_layer)
                        attn_weights[:, f_head, :, :] = attn_weights[:, f_head, :, :] / module.fix_temperature[idx]
                    else:
                        tri = torch.tril(torch.ones((tgt_len, src_len), dtype=attn_weights.dtype, device=attn_weights.device))
                        uni = tri / tri.sum(dim=-1, keepdim=True)
                        attn_weights[:, f_head, :, :] = uni.unsqueeze(0).expand(bsz, tgt_len, src_len)

    # 8) Value との積で出力
    return torch.matmul(attn_weights, value)


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    enable_gqa: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # 1) GQA 前処理
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    # 2) Mask の整形
    causal_mask = attention_mask
    if attention_mask is not None and causal_mask.ndim == 4:
        causal_mask = causal_mask[:, :, :, : key.size(-2)]

    # 3) contiguous
    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()

    # 4) is_causal の決定
    if is_causal is None:
        is_causal = (query.size(-2) > 1) and (causal_mask is None)
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = bool(is_causal.item())

    # 5) SDPA 呼び出し
    attn_output = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
        module=module,
    )

    # 6) 出力整形
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
