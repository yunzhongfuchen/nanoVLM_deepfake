# tests/test_convnextv2.py
import torch
import torch.nn as nn
from models.convnext_v2 import ConvNeXtV2
from models.config import VLMConfig


def test_convnextv2():
    print("🧪 Starting ConvNeXtV2 Test Suite\n")

    # Step 1: Create configuration
    print("[1/5] 🔧 Creating configuration...")
    cfg = VLMConfig()
    print(f"   ✔️ Image size: {cfg.convnext_img_size}")
    print(f"   ✔️ Hidden dim: {cfg.convnext_hidden_dim}\n")

    # Step 2: Build random-initialized model
    print("[2/5] 🧱 Building random-initialized model...")
    try:
        model_random = ConvNeXtV2(cfg, load_backbone=False)
    except Exception as e:
        raise RuntimeError("❌ Failed to create model") from e

    total_params = sum(p.numel() for p in model_random.parameters())
    print(f"   ✔️ Model created with {total_params:,} parameters")
    print(f"   ✔️ Output token count: {model_random.num_patches} (grid: {model_random.feat_h}x{model_random.feat_w})\n")

    # Step 3: Forward pass (random init)
    print("[3/5] 🔄 Testing forward pass (random weights)...")
    x = torch.randn(2, 3, 224, 224)
    try:
        with torch.no_grad():
            out_random = model_random(x)
    except Exception as e:
        raise RuntimeError("❌ Forward pass failed") from e

    expected_shape = (2, model_random.num_patches, cfg.convnext_hidden_dim)
    assert out_random.shape == expected_shape, \
        f"Expected {expected_shape}, got {out_random.shape}"

    print(f"   ✔️ Output shape OK: {out_random.shape}")
    print(f"   ✔️ Mean abs value: {out_random.abs().mean().item():.4f}")
    print(f"   ✔️ Std: {out_random.std().item():.4f}\n")

    # Step 4: Test from_pretrained
    print("[4/5] ⬇️ Testing from_pretrained interface...")
    try:
        model_pretrained = ConvNeXtV2.from_pretrained(cfg)
    except Exception as e:
        print(f"❌ Error in from_pretrained: {str(e)}")
        raise

    print(f"   ✔️ Pre-trained model loaded successfully.\n")

    # Step 5: Forward pass (pre-trained)
    print("[5/5] 🧪 Testing forward pass (pre-trained weights)...")
    try:
        with torch.no_grad():
            out_pretrained = model_pretrained(x)
    except Exception as e:
        raise RuntimeError("❌ Forward pass failed in pre-trained model") from e

    assert out_pretrained.shape == expected_shape, \
        f"Expected {expected_shape}, got {out_pretrained.shape}"

    print(f"   ✔️ Output shape OK: {out_pretrained.shape}")
    print(f"   ✔️ Mean abs value: {out_pretrained.abs().mean().item():.4f}")

    # Check first layer norm (should be non-zero)
    first_norm = model_pretrained.backbone.stem[0].weight.data.norm().item()
    print(f"   💡 First conv weight L2 norm: {first_norm:.4f}")
    assert first_norm > 1e-5, "❌ Backbone appears untrained"

    # Final success message
    print("\n🎉 All tests passed!")
    print(f"✅ ConvNeXtV2 outputs native shape: [B, {model_random.num_patches}, {cfg.convnext_hidden_dim}]")
    print(f"💡 You can now fuse it with ViT using ModalityProjector or fusion layer.")


if __name__ == "__main__":
    test_convnextv2()
