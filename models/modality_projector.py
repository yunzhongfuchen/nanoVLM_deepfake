# Modality Projection from Vision to Language
import torch.nn as nn

class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor**2)
        self.output_dim = cfg.lm_hidden_dim
        self.scale_factor = cfg.mp_pixel_shuffle_factor

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq # Sequence length must be a perfect square for pixel shuffle
        assert seq_root % self.scale_factor == 0 # Sequence root must be divisible by scale factor

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor
        
        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2)
        
        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)

        return x

# models/fusion_layer.py
import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """
    A trainable, reusable cross-attention fusion module.
    
    - Maintains learnable parameters (Linear layers)
    - Can be trained with backprop
    - Compatible with your VLM pipeline
    """
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.vit_hidden_dim  # e.g., 768
        self.num_heads = getattr(cfg, 'fusion_num_heads', 8)
        assert self.embed_dim % self.num_heads == 0

        # Learnable projection layers
        self.proj_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_v = nn.Linear(self.embed_dim, self.embed_dim)

        # Dropout for training stability
        self.dropout = nn.Dropout(getattr(cfg, 'fusion_dropout', 0.1))

        # Initialize properly
        self._init_weights()

    def _init_weights(self):
        """Initialize projections"""
        for m in [self.proj_q, self.proj_k, self.proj_v]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, vit_features, cn_features):
        """
        Fuse ViT and CNN features using cross-attention.

        Args:
            vit_features: [B, 196, 768] ← Query path
            cn_features:  [B, 49, 768]  ← Key/Value source

        Returns:
            fused: [B, 196, 768] — same shape as input
        """
        B, N_vit, D = vit_features.shape
        _, N_cn, _ = cn_features.shape

        # Project to unified space
        q = self.proj_q(vit_features)   # [B,196,D]
        k = self.proj_k(cn_features)    # [B,49,D]
        v = self.proj_v(cn_features)    # [B,49,D]

        # Reshape for multi-head attention
        q = q.view(B, N_vit, self.num_heads, D // self.num_heads).transpose(1, 2)  # [B,h,N,d]
        k = k.view(B, N_cn, self.num_heads, D // self.num_heads).transpose(1, 2)   # [B,h,N,d]
        v = v.view(B, N_cn, self.num_heads, D // self.num_heads).transpose(1, 2)   # [B,h,N,d]

        # Scaled dot-product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_vit, D)

        # Residual connection
        fused = vit_features + attn_output  # [B,196,768]

        return fused