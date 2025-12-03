import argparse
import torch
from PIL import Image

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate nanoVLM on SID validation set")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a local checkpoint (directory or safetensors/pth). If omitted, we pull from HF."
    )
    parser.add_argument(
        "--hf_model", type=str, default="lusxvr/nanoVLM-222M",
        help="HuggingFace repo ID to download from if --checkpoint is not set."
    )
    parser.add_argument(
        "--dataset", type=str, default="saberzl/SID_Set",
        help="Dataset name to load. Must have a 'validation' split."
    )
    parser.add_argument(
        "--split", type=str, default="validation",
        help="Dataset split to evaluate on (e.g., 'validation', 'test')"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=32,
        help="Maximum number of tokens to generate per answer"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of samples for quick testing (e.g., 100)"
    )
    parser.add_argument(
        "--print_examples", type=int, default=5,
        help="Number of (pred, gt) examples to print"
    )
    return parser.parse_args()


def normalize_answer(text):
    """将模型输出归一化为三类标签"""
    t = text.lower()
    if 'real' in t and 'synthetic' not in t and 'tampered' not in t:
        return 'real'
    elif 'tampered' in t or 'manipulated' in t or 'edited' in t or 'spliced' in t:
        return 'tampered'
    elif 'synthetic' in t or 'generated' in t or 'ai' in t or 'fake' in t or 'full' in t:
        return 'full_synthetic'
    else:
        # fallback: keyword matching
        if 'real' in t:
            return 'real'
        if any(k in t for k in ['synth', 'fake', 'ai', 'gen', 'computer']):
            return 'full_synthetic'
        if any(k in t for k in ['tamper', 'edit', 'manipulat', 'alter']):
            return 'tampered'
        return 'unknown'


def extract_gt_label(label_id):
    """SID dataset 中 label: 0=real, 1=full synthetic, 2=tampered"""
    label_map = {
        0: 'real',
        1: 'full_synthetic',
        2: 'tampered'
    }
    return label_map.get(label_id, 'unknown')


def main():
    args = parse_args()

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    source = args.checkpoint if args.checkpoint else args.hf_model
    print(f"Loading weights from: {source}")
    model = VisionLanguageModel.from_pretrained(source).to(device)
    model.eval()

    # Tokenizer & image processor
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    # Prompt template
    question = "Is this image real, full synthetic or tampered?"
    prompt = f"Question: {question} Answer:"

    # Encode prompt once
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Load SID dataset
    print(f"Loading dataset '{args.dataset}' split '{args.split}'...")
    dataset = load_dataset(args.dataset)[args.split]
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    print(f"Starting evaluation on {len(dataset)} samples...")

    total = 0
    correct = 0
    print_examples = []

    with torch.no_grad():
        for i, item in enumerate(dataset):
            # Get image and label
            try:
                image = item["image"]
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image)
                image = image.convert("RGB")
                img_tensor = image_processor(image).unsqueeze(0).to(device)  # [1, C, H, W]

                label = item["label"]
                gt_label = extract_gt_label(label)
                if gt_label == 'unknown':
                    continue

                # Generate
                gen_ids = model.generate(
                    input_ids, 
                    img_tensor, 
                    attention_mask, 
                    max_new_tokens=args.max_new_tokens  # 假设你的 generate 支持这个参数
                )
                pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                pred_label = normalize_answer(pred_text)

                # Record for printing
                if len(print_examples) < args.print_examples:
                    print_examples.append({
                        'gt': gt_label,
                        'pred': pred_label,
                        'raw': pred_text
                    })

                if pred_label == gt_label:
                    correct += 1
                total += 1

                if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
                    print(f"Processed {i+1}/{len(dataset)} | Current Acc: {correct/total:.3f}")

            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue

    # Final results
    accuracy = correct / total if total > 0 else 0
    print("\n" + "="*50)
    print(f"✅ Evaluation Complete")
    print(f"Total Samples: {total}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("="*50)

    # Print some examples
    print(f"\n🔍 First {len(print_examples)} predictions:")
    for idx, ex in enumerate(print_examples):
        status = "✅" if ex['gt'] == ex['pred'] else "❌"
        print(f"{idx+1}. {status} GT={ex['gt']}, Pred={ex['pred']} | '{ex['raw']}'")


if __name__ == "__main__":
    main()
