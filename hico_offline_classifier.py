import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List

from models.dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
from datasets.hico_text_label import hico_text_label, hico_obj_text_label, hico_unseen_index


''' Usage:
python hico_offline_classifier.py \
    --dinotxt_weights <path_to_dinov3_text_head_and_vision_head_weights> \
    --backbone_weights <path_to_dinov3_backbone_weights> \
    --bpe_path_or_url <path_or_url_to_bpe_vocab> \
    --save_dir params/hico

'''


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for DINOv3 classifier embeddings saving.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Save DINOv3 classifier embeddings for HOI detection (train/eval splits)."
    )
    parser.add_argument(
        "--dinotxt_weights", type=str,
        default="dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
        help="Path to DINOv3 text head + vision head weights."
    )
    parser.add_argument(
        "--backbone_weights", type=str,
        default="dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        help="Path to DINOv3 backbone weights."
    )
    parser.add_argument(
        "--bpe_path_or_url", type=str,
        default="bpe_simple_vocab_16e6.txt.gz",
        help="Path or URL to BPE vocabulary for DINOv3 tokenizer."
    )
    parser.add_argument(
        "--save_dir", type=str,
        default="classifier_weights",
        help="Directory to save classifier embedding files."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for text embedding generation to prevent GPU OOM."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for text embedding generation, e.g. cuda or cpu. Defaults to cuda when available."
    )
    return parser.parse_args()


def _clean_prompt(text: str) -> str:
    return " ".join(text.split())


def _strip_photo_prefix(text: str) -> str:
    prefix = "a photo of "
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def _mask_object_phrase(object_phrase: str, mask_token: str) -> str:
    words = object_phrase.split()
    if words and words[0] in {"a", "an", "the"}:
        return f"{words[0]} {mask_token}"
    return mask_token


def _split_hico_phrase(text: str, object_phrase: str) -> Tuple[str, str]:
    clean_text = _clean_prompt(_strip_photo_prefix(text))
    object_phrase = _clean_prompt(_strip_photo_prefix(object_phrase or "object"))
    person_pos = clean_text.find("person")
    object_pos = clean_text.rfind(object_phrase) if object_phrase else -1
    if person_pos < 0 or object_pos < 0 or object_pos <= person_pos:
        return clean_text.replace("person", "", 1).strip() or clean_text, object_phrase
    relation_phrase = clean_text[person_pos + len("person"):object_pos].strip()
    return relation_phrase or "interact with", object_phrase


def build_hico_semantic_slot_prompts(
    hoi_text_label: Dict[Any, str],
    obj_text_label: list,
    mask_token: str = "<MASK>",
) -> Tuple[List[List[str]], List[List[str]]]:
    obj_text_lookup = {idx: text for idx, text in obj_text_label}
    masked_slots, prior_slots = [], []
    for hoi_key, text in hoi_text_label.items():
        obj_id = hoi_key[1] if isinstance(hoi_key, tuple) and len(hoi_key) > 1 else None
        relation_phrase, object_phrase = _split_hico_phrase(text, obj_text_lookup.get(obj_id, "object"))
        full_prompt = _clean_prompt(_strip_photo_prefix(text))
        masked_object = _mask_object_phrase(object_phrase, mask_token)

        masked_slots.append([
            _clean_prompt(f"person {mask_token} {object_phrase}"),
            _clean_prompt(f"person {relation_phrase} {masked_object}"),
            _clean_prompt(f"person with {mask_token} {relation_phrase} {object_phrase}"),
            _clean_prompt(f"person {relation_phrase} {object_phrase} in {mask_token}"),
        ])
        prior_slots.append([
            _clean_prompt(f"spatial relation evidence for {full_prompt}: person {relation_phrase} {object_phrase}"),
            _clean_prompt(f"object evidence for {full_prompt}: {object_phrase}"),
            _clean_prompt(f"human pose evidence for {full_prompt}: pose while {relation_phrase} {object_phrase}"),
            _clean_prompt(f"scene context evidence for {full_prompt}"),
        ])
    return masked_slots, prior_slots


def build_hico_masked_prompt_variants(
    hoi_text_label: Dict[Any, str],
    obj_text_label: list,
    mask_token: str = "<MASK>",
) -> List[List[str]]:
    masked_slots, _ = build_hico_semantic_slot_prompts(hoi_text_label, obj_text_label, mask_token=mask_token)
    return masked_slots


def encode_slot_prompt_embeddings(
    slot_prompts: List[List[str]],
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    slot_counts = {len(item) for item in slot_prompts}
    if len(slot_counts) != 1:
        raise ValueError(f"All HOI semantic slot prompt groups must have the same length, got {slot_counts}")
    num_slots = slot_counts.pop()
    flat_texts = [text for item in slot_prompts for text in item]
    flat_embeddings = encode_texts(
        flat_texts, model=model, tokenizer=tokenizer, device=device, batch_size=batch_size
    )
    return flat_embeddings.view(len(slot_prompts), num_slots, -1)


def encode_texts(
    texts: List[str],
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start: start + batch_size]
        text_inputs = torch.cat([tokenizer.tokenize(text) for text in batch_texts])
        with torch.inference_mode():
            text_embedding = model.encode_text(text_inputs.to(device)).float()
            text_embedding = F.normalize(text_embedding, p=2, dim=-1)
        embeddings.append(text_embedding.cpu())
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0)


def encode_masked_prompt_embeddings(
    masked_variants: List[List[str]],
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    variant_counts = [len(item) for item in masked_variants]
    flat_texts = [text for item in masked_variants for text in item]
    flat_embeddings = encode_texts(
        flat_texts, model=model, tokenizer=tokenizer, device=device, batch_size=batch_size
    )

    masked_embeddings = []
    offset = 0
    for count in variant_counts:
        item_embedding = flat_embeddings[offset: offset + count]
        masked_embeddings.append(F.normalize(item_embedding.mean(dim=0), p=2, dim=-1))
        offset += count
    return torch.stack(masked_embeddings, dim=0)

def init_classifier_with_dino(
    del_unseen: bool,
    zero_shot_type: str,
    hoi_text_label: Dict[Any, str],
    obj_text_label: list,
    unseen_index: Dict[str, list],
    model: torch.nn.Module,
    tokenizer: Any,
    device: str,
    batch_size: int = 64,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[Any, str],
    torch.Tensor,
]:
    """
    Initialize classifier text embeddings for HOI and object labels using DINOv3 text encoder.

    Args:
        del_unseen (bool): If True, filter out unseen HOI classes for training embeddings.
        zero_shot_type (str): Key to select unseen indices.
        hoi_text_label (Dict[Any, str]): Mapping from HOI label indices to label text.
        obj_text_label (list): List of tuples (idx, text) for object labels.
        unseen_index (Dict[str, list]): Mapping from zero-shot type to unseen indices.
        model (torch.nn.Module): DINOv3 model with text encoder.
        tokenizer (Any): DINOv3 tokenizer object.
        device (str): Device string ('cuda' or 'cpu').

    Returns:
        Tuple containing:
            - hoi_embedding_train (torch.Tensor): HOI text embeddings for training. Shape [K_train, D].
            - hoi_embedding_eval (torch.Tensor): HOI text embeddings for evaluation (full set). Shape [K_eval, D].
            - obj_text_embedding_eval (torch.Tensor): Object text embeddings for evaluation. Shape [K_obj, D].
            - hoi_masked_embedding_train (torch.Tensor): Masked HOI text embeddings for training. Shape [K_train, D].
            - hoi_masked_embedding_eval (torch.Tensor): Masked HOI text embeddings for evaluation. Shape [K_eval, D].
            - hoi_masked_slot_embedding_train (torch.Tensor): Masked slot embeddings [K_train, 4, D].
            - hoi_masked_slot_embedding_eval (torch.Tensor): Masked slot embeddings [K_eval, 4, D].
            - hoi_semantic_prior_train (torch.Tensor): Semantic prior bank [K_train, 4, D].
            - hoi_semantic_prior_eval (torch.Tensor): Semantic prior bank [K_eval, 4, D].
            - hoi_text_label_train (Dict[Any, str]): Filtered HOI label dict for training.
            - obj_text_inputs (torch.Tensor): Tokenized object label texts. Shape [K_obj, L].
    """
    # Tokenize all HOI label texts (for evaluation)
    eval_texts = [hoi_text_label[id] for id in hoi_text_label.keys()]

    # Filter HOI labels for training if del_unseen is True
    if del_unseen and unseen_index is not None:
        unseen_index_list = unseen_index.get(zero_shot_type, [])
        hoi_text_label_train = {
            k: hoi_text_label[k]
            for idx, k in enumerate(hoi_text_label.keys())
            if idx not in unseen_index_list
        }
    else:
        hoi_text_label_train = hoi_text_label.copy()

    # Tokenize HOI label texts for training
    train_texts = [hoi_text_label_train[id] for id in hoi_text_label_train.keys()]

    # Tokenize object label texts
    obj_text_inputs = torch.cat([tokenizer.tokenize(obj_text[1]) for obj_text in obj_text_label])

    hoi_masked_slots_eval, hoi_prior_slots_eval = build_hico_semantic_slot_prompts(hoi_text_label, obj_text_label)
    hoi_masked_slots_train, hoi_prior_slots_train = build_hico_semantic_slot_prompts(
        hoi_text_label_train, obj_text_label
    )

    # Encode text embeddings with DINOv3 in batches.
    hoi_embedding_eval = encode_texts(eval_texts, model=model, tokenizer=tokenizer, device=device, batch_size=batch_size)
    hoi_embedding_train = encode_texts(train_texts, model=model, tokenizer=tokenizer, device=device, batch_size=batch_size)
    obj_texts = [obj_text[1] for obj_text in obj_text_label]
    obj_text_embedding_eval = encode_texts(
        obj_texts, model=model, tokenizer=tokenizer, device=device, batch_size=batch_size
    )
    hoi_masked_slot_embedding_train = encode_slot_prompt_embeddings(
        hoi_masked_slots_train, model=model, tokenizer=tokenizer, device=device, batch_size=batch_size
    )
    hoi_masked_slot_embedding_eval = encode_slot_prompt_embeddings(
        hoi_masked_slots_eval, model=model, tokenizer=tokenizer, device=device, batch_size=batch_size
    )
    hoi_semantic_prior_train = encode_slot_prompt_embeddings(
        hoi_prior_slots_train, model=model, tokenizer=tokenizer, device=device, batch_size=batch_size
    )
    hoi_semantic_prior_eval = encode_slot_prompt_embeddings(
        hoi_prior_slots_eval, model=model, tokenizer=tokenizer, device=device, batch_size=batch_size
    )
    hoi_masked_embedding_train = F.normalize(hoi_masked_slot_embedding_train.mean(dim=1), p=2, dim=-1)
    hoi_masked_embedding_eval = F.normalize(hoi_masked_slot_embedding_eval.mean(dim=1), p=2, dim=-1)

    return (
        hoi_embedding_train,
        hoi_embedding_eval,
        obj_text_embedding_eval,
        hoi_masked_embedding_train,
        hoi_masked_embedding_eval,
        hoi_masked_slot_embedding_train,
        hoi_masked_slot_embedding_eval,
        hoi_semantic_prior_train,
        hoi_semantic_prior_eval,
        hoi_text_label_train,
        obj_text_inputs,
    )

def save_classifier_eval(
    hoi_embedding_eval: torch.Tensor,
    obj_text_embedding_eval: torch.Tensor,
    hoi_masked_embedding_eval: torch.Tensor,
    hoi_masked_slot_embedding_eval: torch.Tensor,
    hoi_semantic_prior_eval: torch.Tensor,
    save_path: Path
) -> None:
    """
    Save classifier eval stage embeddings (HOI + object) to a single file.

    Args:
        hoi_embedding_eval (torch.Tensor): HOI text embeddings [K_eval, D].
        obj_text_embedding_eval (torch.Tensor): Object text embeddings [K_obj, D].
        hoi_masked_embedding_eval (torch.Tensor): Masked HOI text embeddings [K_eval, D].
        hoi_masked_slot_embedding_eval (torch.Tensor): Masked slot embeddings [K_eval, 4, D].
        hoi_semantic_prior_eval (torch.Tensor): Semantic prior bank [K_eval, 4, D].
        save_path (Path): Output .pt file path.
    """
    torch.save(
        {
            "hoi_embedding_eval": hoi_embedding_eval.cpu(),
            "hoi_embedding_masked_eval": hoi_masked_embedding_eval.cpu(),
            "hoi_masked_slot_embedding_eval": hoi_masked_slot_embedding_eval.cpu(),
            "hoi_semantic_prior_eval": hoi_semantic_prior_eval.cpu(),
            "obj_text_embedding_eval": obj_text_embedding_eval.cpu()
        },
        str(save_path)
    )

def save_classifier_train(
    hoi_embedding_train: torch.Tensor,
    hoi_masked_embedding_train: torch.Tensor,
    hoi_masked_slot_embedding_train: torch.Tensor,
    hoi_semantic_prior_train: torch.Tensor,
    save_path: Path
) -> None:
    """
    Save classifier train stage embeddings (HOI, filtered by zero-shot split) to a single file.

    Args:
        hoi_embedding_train (torch.Tensor): HOI text embeddings for training [K_train, D].
        hoi_masked_embedding_train (torch.Tensor): Masked HOI text embeddings for training [K_train, D].
        hoi_masked_slot_embedding_train (torch.Tensor): Masked slot embeddings [K_train, 4, D].
        hoi_semantic_prior_train (torch.Tensor): Semantic prior bank [K_train, 4, D].
        save_path (Path): Output .pt file path.
    """
    torch.save(
        {
            "hoi_embedding_train": hoi_embedding_train.cpu(),
            "hoi_embedding_masked_train": hoi_masked_embedding_train.cpu(),
            "hoi_masked_slot_embedding_train": hoi_masked_slot_embedding_train.cpu(),
            "hoi_semantic_prior_train": hoi_semantic_prior_train.cpu()
        },
        str(save_path)
    )

def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load DINOv3 model and tokenizer
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
        dinotxt_weights=args.dinotxt_weights,
        backbone_weights=args.backbone_weights,
        bpe_path_or_url=args.bpe_path_or_url
    )
    model = model.to(device)
    model.eval()

    # Load HOI and object labels, unseen index
    hoi_text_label = hico_text_label
    obj_text_label = hico_obj_text_label
    unseen_index = hico_unseen_index

    # Compute eval embeddings (these do not depend on del_unseen or zero_shot_type)
    (
        _,
        hoi_embedding_eval,
        obj_text_embedding_eval,
        _,
        hoi_masked_embedding_eval,
        _,
        hoi_masked_slot_embedding_eval,
        _,
        hoi_semantic_prior_eval,
        _,
        _,
    ) = init_classifier_with_dino(
        del_unseen=False,
        zero_shot_type="default",
        hoi_text_label=hoi_text_label,
        obj_text_label=obj_text_label,
        unseen_index=unseen_index,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
    )

    # Save eval embeddings (HOI + object) in a single file
    eval_path = save_dir / "classifier_eval.pt"
    save_classifier_eval(
        hoi_embedding_eval,
        obj_text_embedding_eval,
        hoi_masked_embedding_eval,
        hoi_masked_slot_embedding_eval,
        hoi_semantic_prior_eval,
        eval_path,
    )
    print(f"Saved eval classifier embeddings to {eval_path}")

    # Define zero-shot splits; you can add or modify splits as needed
    zero_shot_splits = [
        ("default", False),
        ("rare_first", True),
        ("non_rare_first", True),
        ("unseen_object", True),
        ("unseen_verb", True),
    ]

    # For each split, save its classifier train embedding
    for zero_shot_type, del_unseen in zero_shot_splits:
        print(f"Processing train split: zero_shot_type={zero_shot_type}, del_unseen={del_unseen}")
        (
            hoi_embedding_train,
            _,
            _,
            hoi_masked_embedding_train,
            _,
            hoi_masked_slot_embedding_train,
            _,
            hoi_semantic_prior_train,
            _,
            hoi_text_label_train,
            _,
        ) = init_classifier_with_dino(
            del_unseen=del_unseen,
            zero_shot_type=zero_shot_type,
            hoi_text_label=hoi_text_label,
            obj_text_label=obj_text_label,
            unseen_index=unseen_index,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
        )
        print(f"  hoi_embedding_train shape: {hoi_embedding_train.shape}")
        print(f"  hoi_text_label_train count: {len(hoi_text_label_train)}")

        train_path = save_dir / f"classifier_{zero_shot_type}.pt"
        save_classifier_train(
            hoi_embedding_train,
            hoi_masked_embedding_train,
            hoi_masked_slot_embedding_train,
            hoi_semantic_prior_train,
            train_path,
        )
        print(f"Saved train classifier embeddings to {train_path}")

if __name__ == "__main__":
    main()
