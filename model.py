import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(mean_sq + self.eps)
        return self.scale * x_norm

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU: (Swish(xW) * xV)W_out
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.dropout(self.w3(hidden))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        # x: [B, T, C]
        t = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, freqs):
    # x: [B, T, H, D]
    # freqs: [T, D] -> [1, T, 1, D]
    freqs = freqs[None, :x.shape[1], None, :]
    return (x * freqs.cos()) + (rotate_half(x) * freqs.sin())

class MultiHeadLatentAttention(nn.Module):
    """
    MLA with RoPE and Flash Attention (Scaled Dot Product).
    """
    def __init__(self, d_model, num_heads, d_latent=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_latent = d_latent if d_latent is not None else d_model // 4 
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        # QK Norm (Stability for large models/FP16)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.q_down = nn.Linear(d_model, self.d_latent, bias=False)
        self.kv_down = nn.Linear(d_model, self.d_latent, bias=False)
        
        self.q_up = nn.Linear(self.d_latent, d_model, bias=False)
        self.k_up = nn.Linear(self.d_latent, d_model, bias=False)
        self.v_up = nn.Linear(self.d_latent, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # 1. Compress to Latent
        latent_q = self.q_down(x) 
        latent_kv = self.kv_down(x) 
        
        # 2. Expand to Heads
        q = self.q_up(latent_q).view(B, T, self.num_heads, self.head_dim)
        k = self.k_up(latent_kv).view(B, T, self.num_heads, self.head_dim)
        v = self.v_up(latent_kv).view(B, T, self.num_heads, self.head_dim)
        
        # 3. Apply QK Norm (Before RoPE)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # 4. Apply RoPE (Rotary Embeddings)
        freqs = self.rotary_emb(q) # Generate freqs based on T
        q = apply_rotary_pos_emb(q, freqs)
        k = apply_rotary_pos_emb(k, freqs)

        # 4. Flash Attention (PyTorch 2.0)
        # Transpose for Flash: [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use efficient SDP Kernel with Causal=True
        # This automatically applies the triangular mask AND uses the efficient kernel
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0 if not self.training else 0.1)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(out)

class RecursiveBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadLatentAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN (RMS)
        # Note: We pass None for mask to let SDPA handle causal logic if implemented,
        # or we implement is_causal=True inside Attention.
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class HierarchicalReasoningModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_reasoning_tokens=5, dropout=0.1, max_len=2048):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_reasoning_tokens = num_reasoning_tokens
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # self.position_embedding = removed (Using RoPE)
        
        self.reasoning_tokens = nn.Parameter(torch.randn(1, num_reasoning_tokens, d_model))
        
        self.shared_block = RecursiveBlock(d_model, num_heads, d_ff, dropout)
        
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
        
        # Gradient Checkpointing Support
        self.use_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        
        x = self.token_embedding(idx)
        # No absolute pos embedding needed -> RoPE handles it in Attention
        
        r_tokens = self.reasoning_tokens.expand(B, -1, -1)
        x = torch.cat((r_tokens, x), dim=1) 
        
        # Causal Mask for Flash Attention
        # Note: F.sdpa handles Is_Causal=True efficiently.
        # But we have reasoning tokens (Prefix).
        # We need a custom mask: Reasoning can attend to Reasoning. Text to Text+Reasoning.
        # Simple Causal Mask works fine for this: [R1, R2, T1, T2]
        # T2 attends to T1, R1, R2.
        # So standard Causal Mask covers it.
        
        # However, Flash Attention on T4 requires 'attn_mask' to be None for is_causal optimization usually.
        # Since we just want simple causal, we can let specific implementations handle it 
        # OR just pass the boolean mask. 
        # For compatibility with our RoPE implementation, we rely on the implementation.
        
        total_seq_len = x.size(1)
        
        # Logic: If using SDPA, we can pass is_causal=True if the mask is purely causal.
        # Our concat prepends tokens, so strictly speaking it IS causal sequence.
        
        for _ in range(self.num_layers):
            if self.use_checkpointing and self.training:
                 # Checkpointing saves VRAM by re-computing forward pass
                 # Requires dummy variable for generalized checkpointing sometimes
                 x = torch.utils.checkpoint.checkpoint(self.shared_block, x, use_reentrant=False)
            else:
                 x = self.shared_block(x)
            
        x = self.ln_f(x)
        text_out = x[:, self.num_reasoning_tokens:, :] # Extract original tokens
        logits = self.lm_head(text_out)
        
        # Scale logic to prevent initial confident predictions
        # This keeps loss around 10.0 instead of 100.0
        logits = logits * 0.1

        loss = None
        if targets is not None:
            # Shift so that we predict the NEXT token
            # Logits: [B, T, V] -> [B, T-1, V] (Remove last prediction)
            # Targets: [B, T] -> [B, T-1] (Remove first token)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
            
        return logits, loss
