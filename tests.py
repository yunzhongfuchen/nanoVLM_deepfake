# test_cls_token_extraction.py
# 完整测试 VisionLanguageModel 中 [CLS] token 的提取逻辑

import torch
import os
from models.vision_language_model import VisionLanguageModel
from models.config import VLMConfig
from transformers import AutoTokenizer

print("🧪 开始测试：[CLS] token 提取逻辑（已修复 pad_token 问题）")

# ==================================================
# 1. 加载配置与 tokenizer
# ==================================================

cfg = VLMConfig()
tokenizer_name = cfg.lm_tokenizer  # 'HuggingFaceTB/cosmo2-tokenizer'

print(f"Loading tokenizer: {tokenizer_name}")
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
except Exception as e:
    print(f"❌ 无法加载 tokenizer，请检查网络或缓存")
    raise e

# --------------------------------------------------
# ✅ 关键修复：设置 pad_token
# --------------------------------------------------
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"🔧 已设置 pad_token = eos_token ('{tokenizer.eos_token}')")
    else:
        # 如果连 eos_token 都没有，就添加一个
        tokenizer.add_special_tokens({'eos_token': '</s>'})
        tokenizer.pad_token = tokenizer.eos_token
        print(f"🔧 已补充 eos_token 和 pad_token (token='</s>')")

print(f"✅ pad_token = '{tokenizer.pad_token}', id={tokenizer.pad_token_id}")

# --------------------------------------------------
# ✅ 确保 [CLS] token 存在
# --------------------------------------------------
if "[CLS]" not in tokenizer.get_vocab():
    num_added = tokenizer.add_tokens(["[CLS]"])
    print(f"🟢 成功添加 [CLS] token (共新增 {num_added} 个 token)")
else:
    print("🟢 [CLS] 已存在于词汇表中")

# 更新 vocab size 到 config（模拟 resize 前状态）
original_vocab_size = len(tokenizer)

# ==================================================
# 2. 创建模型实例（不加载 backbone 权重）
# ==================================================

print("\n🧠 创建 VisionLanguageModel 实例...")
model = VisionLanguageModel(cfg, load_backbone=False)  # 无需加载预训练权重
model.decoder.resize_token_embeddings(len(tokenizer))
model.eval()

# 🔁 如果你已经实现了 resize_token_embeddings，请取消注释以下两行：
# print(f"🔄 调用 decoder.resize_token_embeddings({len(tokenizer)})")
# model.decoder.resize_token_embeddings(len(tokenizer))

print(f"✅ 模型创建成功")
print(f"   分类头结构: {model.classifier}")

print("\n" + "="*60)
print("1️⃣ 测试 forward 函数中的 [CLS] 提取")
print("="*60)

# ==================================================
# 构造测试输入
# ==================================================

texts = [
    "[CLS] A cat is sitting on the grass.",
    "[CLS] An urban cityscape with tall buildings at night."
]

# 使用 tokenizer 编码，并启用 padding/truncation
inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512  # 避免 warning
)

input_ids = inputs["input_ids"]         # [2, T]
attention_mask = inputs["attention_mask"]  # [2, T]

# ==================================================
# 打印并验证输入格式
# ==================================================

print("📝 输入文本:")
for i, text in enumerate(texts):
    print(f"  [{i}] {text}")

print(f"\n🔢 input_ids.shape: {tuple(input_ids.shape)}")
print(f"第一个样本的第一个 token ID: {input_ids[0, 0].item()}")
print(f"[CLS] 的 token ID: {tokenizer.convert_tokens_to_ids('[CLS]')}")

assert input_ids[0, 0].item() == tokenizer.convert_tokens_to_ids("[CLS]") and \
       input_ids[1, 0].item() == tokenizer.convert_tokens_to_ids("[CLS]"), \
    "❌ 错误：输入未以 [CLS] 开头"

print("✅ 所有输入均以 [CLS] 开头 ✔️")

# ==================================================
# 前向传播测试
# ==================================================

images = torch.randn(2, 3, 224, 224)  # B=2, C=3, H=224, W=224

with torch.no_grad():
    lm_logits, total_loss, class_logits = model(
        input_ids=input_ids,
        image=images,
        attention_mask=attention_mask,
        targets=input_ids.clone(),      # mock target for gen loss
        targets_cls=torch.tensor([0, 2])  # fake labels for classification
    )

print(f"\n🔍 输出形状:")
print(f"  lm_logits.shape     : {tuple(lm_logits.shape)}")
print(f"  total_loss          : {total_loss.item():.4f}")
print(f"  class_logits.shape  : {tuple(class_logits.shape)} → 应为 (2, 3)")

assert class_logits.shape == (2, 3), "分类 logits 形状错误"
print("✅ class_logits 形状正确 ✅")

# ==================================================
# 手动复现 cls_position 提取过程（调试用）
# ==================================================

image_embd = model.vision_encoder(images)
image_embd = model.MP(image_embd)
img_seq_len = image_embd.size(1)

print(f"\n📊 图像 token 序列长度: {img_seq_len}")
print(f"cls_position = img_seq_len = {img_seq_len}")

token_embd = model.decoder.token_embedding(input_ids)
combined_embd = torch.cat((image_embd, token_embd), dim=1)

# 获取隐藏状态
hidden_states = model.decoder(combined_embd, attention_mask)

# 手动提取 [CLS] 表示
cls_hidden_state = hidden_states[:, img_seq_len:img_seq_len+1, :]  # [2,1,D]
manual_class_logits = model.classifier(cls_hidden_state).squeeze(1)

diff = (manual_class_logits - class_logits).abs().max()
print(f"手动计算 vs 模型内部输出最大差异: {diff:.6f}")
assert diff < 1e-5, "推理结果不一致"

print("🟢 手动验证通过！[CLS] 提取逻辑完全正确。")
