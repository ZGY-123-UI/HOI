import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from datasets.hico_text_label import hico_text_label, hico_unseen_index


def load_clip_model(model_name: str, device: str):
    try:
        import clip
    except ImportError as exc:
        raise ImportError("Please install OpenAI CLIP first: pip install git+https://github.com/openai/CLIP.git") from exc
    model, _ = clip.load(model_name, device=device)
    model.eval()
    return clip, model


def build_text_items(zero_shot_type: str, del_unseen: bool):
    text_label_ids = list(hico_text_label.keys())
    if del_unseen:
        unseen = set(hico_unseen_index.get(zero_shot_type, []))
        text_label_ids = [key for idx, key in enumerate(text_label_ids) if idx not in unseen]
    texts = [hico_text_label[key] for key in text_label_ids]
    return text_label_ids, texts


def main():
    parser = argparse.ArgumentParser(description="Build HICO CLIP text soft-label matrix.")
    parser.add_argument("--output", default="/media/qdu/2.0T/zgy/projects/SL-HOI/params/hico/clip_soft_label.pt")
    parser.add_argument("--model", default="ViT-B/32")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--zero-shot-type", default="default")
    parser.add_argument("--del-unseen", action="store_true")
    args = parser.parse_args()

    clip, model = load_clip_model(args.model, args.device)
    text_label_ids, texts = build_text_items(args.zero_shot_type, args.del_unseen)

    tokens = clip.tokenize(texts).to(args.device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features.float(), dim=-1)
        sim = text_features @ text_features.t()
        clip_soft_label = torch.softmax(sim / args.temperature, dim=-1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "clip_soft_label": clip_soft_label.cpu(),
            # Stage 1/4: keep normalized CLIP text features so loss-time soft labels
            # and optional union-crop image distillation use the same HOI vocabulary.
            "clip_text_features": text_features.cpu(),
            "text_label_ids": text_label_ids,
            "texts": texts,
            "clip_model": args.model,
            "temperature": args.temperature,
            "zero_shot_type": args.zero_shot_type,
            "del_unseen": args.del_unseen,
        },
        output,
    )
    print(f"Saved CLIP soft label matrix {tuple(clip_soft_label.shape)} to {output}")


if __name__ == "__main__":
    main()