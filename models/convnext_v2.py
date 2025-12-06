# models/convnext_v2.py
import torch
import torch.nn as nn
import timm


class ConvNeXtV2(nn.Module):
    """
    Faithful implementation of ConvNeXtV2-Base.
    Outputs: [B, 49, 768] (native feature map size).
    No forced upsampling to 196. Let downstream handle sequence length.
    """
    def __init__(self, cfg, load_backbone=False):
        super().__init__()
        self.cfg = cfg

        # Load backbone
        print("[ConvNeXtV2] Loading {cfg.convnext_model_type}...")
        self.backbone = timm.create_model(
            cfg.convnext_model_type,
            pretrained=load_backbone,
            num_classes=0,
            global_pool=""
        )

        # Detect output channels and spatial shape via dummy forward
        with torch.no_grad():
            device = next(self.backbone.parameters()).device
            x = torch.randn(1, 3, 224, 224).to(device)
            feat_map = self.backbone(x)
            _, C, H, W = feat_map.shape
            self.feat_h, self.feat_w = H, W
            self.num_patches = H * W  # e.g., 49
            self.in_chans = C

        print(f"[ConvNeXtV2] Native output grid: {H}x{W}={self.num_patches} patches")
        print(f"[ConvNeXtV2] Final channel dim: {self.in_chans}")

        # Project to target embedding dimension
        self.patch_proj = nn.Conv2d(
            in_channels=self.in_chans,
            out_channels=cfg.convnext_hidden_dim,
            kernel_size=1
        )
        self.embd_dim = cfg.convnext_hidden_dim
        self.dropout = nn.Dropout(cfg.convnext_dropout)

        # Position embedding for actual number of tokens
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, self.embd_dim))

        # Initialize only newly added layers
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Parameter) and module.dim() == 3:
            if module.size() == self.position_embedding.size():
                nn.init.normal_(module, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Extract feature map: [B, C, H, W]
        feat_map = self.backbone(x)

        # Project to target dim: [B, D, H, W]
        proj_map = self.patch_proj(feat_map)

        # Flatten to sequence: [B, N, D]
        patches = proj_map.flatten(2).transpose(1, 2)

        # Add position embedding and dropout
        patches = patches + self.position_embedding
        patches = self.dropout(patches)

        return patches  # [B, 49, 768]

    @classmethod
    def from_pretrained(cls, cfg):
        print("Loading from backbone weights")
        model = cls(cfg, load_backbone=True)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Successfully loaded. Total params: {total_params:,}")
        return model
