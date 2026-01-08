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

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) with RMSNorm integration.
    """
    def __init__(self, d_model, num_heads, d_latent=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_latent = d_latent if d_latent is not None else d_model // 4 
        self.scale = self.head_dim ** -0.5

        self.q_down = nn.Linear(d_model, self.d_latent, bias=False)
        self.kv_down = nn.Linear(d_model, self.d_latent, bias=False)
        
        self.q_up = nn.Linear(self.d_latent, d_model, bias=False)
        self.k_up = nn.Linear(self.d_latent, d_model, bias=False)
        self.v_up = nn.Linear(self.d_latent, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        latent_q = self.q_down(x) 
        latent_kv = self.kv_down(x) 
        
        q = self.q_up(latent_q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_up(latent_kv).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(latent_kv).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
             attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(out)

class RecursiveBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadLatentAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN (RMS)
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class HierarchicalReasoningModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_reasoning_tokens=5, dropout=0.1, max_len=2048):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_reasoning_tokens = num_reasoning_tokens
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        self.reasoning_tokens = nn.Parameter(torch.randn(1, num_reasoning_tokens, d_model))
        
        self.shared_block = RecursiveBlock(d_model, num_heads, d_ff, dropout)
        
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        # Init weights properly (Crucial for deep/recursive models)
        self.apply(self._init_weights)

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
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        
        r_tokens = self.reasoning_tokens.expand(B, -1, -1)
        x = torch.cat((r_tokens, x), dim=1) 
        
        total_seq_len = x.size(1)
        mask = torch.tril(torch.ones((total_seq_len, total_seq_len), device=device))
        mask = mask.view(1, 1, total_seq_len, total_seq_len)

        for _ in range(self.num_layers):
            x = self.shared_block(x, mask)
            
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
