"""Optional original compact NLI experiment: no training and no adapter weights.

Requires torch, transformers, safetensors. See docs/NLI_EXPERIMENT.md.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter


def load_base(folder, device):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    config = AutoConfig.from_pretrained(folder, local_files_only=True)
    model = AutoModelForSequenceClassification.from_config(config).float()
    state = load_file(str(folder / "base.safetensors"))
    weights = {}
    for name, tensor in state.items():
        if name == "retained_ids" or name.endswith("__scale"):
            continue
        if name.endswith("__q"):
            parameter = name[:-3]
            weights[parameter] = tensor.float() * state[parameter + "__scale"].float()
        else:
            weights[name] = tensor
    obsolete = "deberta.embeddings.position_ids"
    if obsolete not in model.state_dict():
        weights.pop(obsolete, None)
    model.load_state_dict(weights, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    retained = state["retained_ids"].long()
    unknown = torch.where(retained == tokenizer.unk_token_id)[0].item()
    lookup = torch.full((config.original_vocab_size,), unknown, dtype=torch.long)
    lookup[retained] = torch.arange(len(retained))
    return model.to(device).eval(), tokenizer, lookup, set(retained.tolist())


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("models/deberta"))
    parser.add_argument("--output", type=Path, default=Path("output/nli-experiment.json"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    started = perf_counter()
    model, tokenizer, lookup, retained = load_base(args.base, device)
    cases = [
        (
            "Питание шкафа ТК-101 составляет 220 В.",
            "Питание шкафа ТК-101 составляет 220 В.",
            "entailment",
        ),
        ("Длина кабеля: 18 м.", "Длина кабеля составляет 25 м.", "contradiction"),
        ("Класс защиты корпуса: IP54.", "Класс защиты корпуса должен быть IP65.", "contradiction"),
        ("Класс защиты корпуса: IP54.", "Класс защиты корпуса: IP65.", "contradiction"),
        (
            "Испытания шкафа ТК-101 выполнены.",
            "Отчёт содержит сопротивление изоляции шкафа ТК-101.",
            "neutral",
        ),
    ]
    results = []
    labels = [model.config.id2label[i] for i in range(3)]
    for premise, hypothesis, expected in cases:
        encoded = tokenizer(
            premise, hypothesis, truncation="only_first", max_length=384, return_tensors="pt"
        )
        ids = encoded["input_ids"]
        unknown_pieces = [
            tokenizer.convert_ids_to_tokens(i) for i in ids[0].tolist() if i not in retained
        ]
        inputs = {
            "input_ids": lookup[ids].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }
        with torch.inference_mode():
            probabilities = model(**inputs).logits.softmax(-1)[0].cpu().tolist()
        predicted = labels[max(range(3), key=probabilities.__getitem__)]
        result = {
            "premise": premise,
            "hypothesis": hypothesis,
            "expected": expected,
            "predicted": predicted,
            "probabilities": dict(zip(labels, probabilities, strict=True)),
            "missing_vocabulary_pieces": unknown_pieces,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "device": device,
                "seconds": perf_counter() - started,
                "trained_adapter_used": False,
                "cases": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
