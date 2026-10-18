def attention(query, key, value):
    import torch.nn.functional as functional
    return functional.scaled_dot_product_attention(query, key, value, dropout_p=0., is_causal=False)
