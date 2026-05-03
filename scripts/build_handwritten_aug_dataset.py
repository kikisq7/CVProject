"""Build a pseudo-handwritten dataset from PDMX-Synth's typeset images.

PDMX-Synth ships rendered typeset score images + ABC transcriptions but
**does not** include MusicXML inline, so we cannot feed it through
Smashcima. As a Colab-friendly fallback, we apply heavy stochastic image
augmentation that loosely simulates the visual statistics of handwritten
scans: stroke roughening (morphological + noise), staff-line distortion
(elastic warp), JPEG compression, blur, slight rotation, and contrast
jitter. The transcription column is preserved verbatim, so this dataset
is plug-compatible with ``scripts/train.py``.

The output schema matches PDMX-SmashcimaHW: ``image``, ``transcription``,
``filename``, ``musicxml`` (left empty for compatibility).

This is a *training* convenience -- it lets you fine-tune LEGATO toward
visually-degraded notation without procuring the original PDMX MusicXML
sources. It is **not** a substitute for true handwritten data
(MUSCIMA++ remains the real OOD test set).
"""

import argparse
import io
import os
import random
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Image as HFImage,
    Value,
    load_dataset,
    load_from_disk,
)
from tqdm import tqdm


def _load_source(source: str, slice_per_split: Optional[int]):
    if os.path.isdir(source):
        return load_from_disk(source)
    if slice_per_split is None:
        return load_dataset(source, token=True)

    out = {}
    for split_name in ("train", "val", "validation", "test"):
        try:
            out[split_name] = load_dataset(
                source, split=f"{split_name}[:{slice_per_split}]", token=True
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip-split] {split_name}: {exc}")
    if "validation" in out and "val" not in out:
        out["val"] = out.pop("validation")
    from datasets import DatasetDict as _DD

    return _DD(out)


def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.open(img).convert("RGB")


def _augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Stack several handwriting-like distortions stochastically."""
    # Slight rotation to simulate page skew.
    img = img.rotate(rng.uniform(-3.0, 3.0), resample=Image.BILINEAR,
                     fillcolor=(255, 255, 255), expand=False)

    # Elastic-ish staff line warp via per-row horizontal jitter (cheap).
    arr = np.asarray(img).copy()
    h = arr.shape[0]
    max_shift = int(rng.uniform(1, 4))
    shifts = (np.sin(np.linspace(0, rng.uniform(2, 6) * np.pi, h)) * max_shift).astype(int)
    for y in range(h):
        s = shifts[y]
        if s != 0:
            arr[y] = np.roll(arr[y], s, axis=0)
    img = Image.fromarray(arr)

    # Stroke "ink" roughening: subtract noise from dark pixels so the
    # strokes look less uniform. Equivalent to a tiny dilation+blur path.
    arr = np.asarray(img).astype("int16")
    noise = np.random.normal(0, rng.uniform(8, 18), size=arr.shape).astype("int16")
    arr = (arr + noise).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr)

    # Mild blur.
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.2)))

    # Lower contrast / paper tone: lerp toward a warm gray.
    paper = Image.new("RGB", img.size, (245, 240, 230))
    img = Image.blend(img, paper, rng.uniform(0.05, 0.15))

    # Final JPEG re-compression (handwritten scans are usually JPEG).
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=rng.choice([35, 50, 65]))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_split(
    source_split,
    seed: int,
    max_samples: Optional[int],
    image_format: str,
    images_per_source: int,
) -> Dataset:
    n_source = min(len(source_split), max_samples or len(source_split))

    def generator():
        rng = random.Random(seed)
        for i in tqdm(range(n_source), desc="Augmenting"):
            ex = source_split[i]
            try:
                base = _to_pil(ex["image"])
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {ex.get('filename')}: {exc}")
                continue
            for k in range(images_per_source):
                aug = _augment(base, rng)
                buf = io.BytesIO()
                aug.save(buf, format=image_format)
                yield {
                    "filename": f"{ex.get('filename', f'sample_{i}')}_aug{k}",
                    "transcription": ex.get("transcription", ""),
                    "musicxml": "",
                    "image": {"bytes": buf.getvalue(), "path": None},
                }

    features = Features(
        {
            "filename": Value("string"),
            "transcription": Value("string"),
            "musicxml": Value("string"),
            "image": HFImage(),
        }
    )
    return Dataset.from_generator(generator, features=features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dataset", default="guangyangmusic/PDMX-Synth")
    parser.add_argument("--source_split_slice", type=int, default=None,
                        help="Only download first N rows of each split when loading from the Hub.")
    parser.add_argument("--output_dir", default="datasets/PDMX-SmashcimaHW")
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_val", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--images_per_source", type=int, default=1,
                        help="How many augmented images to emit per source row (>=1).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_format", default="PNG")
    parser.add_argument("--mini_val_size", type=int, default=50)
    parser.add_argument("--mini_test_size", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading source dataset from {args.source_dataset} (slice={args.source_split_slice})")
    source = _load_source(args.source_dataset, args.source_split_slice)
    print(f"Source splits: {list(source.keys())}")

    out = DatasetDict()
    for split_name, cap in (("train", args.max_train), ("val", args.max_val), ("test", args.max_test)):
        if split_name not in source:
            print(f"[warn] no '{split_name}' split, skipping")
            continue
        print(f"\n=== Augmenting split: {split_name} ({len(source[split_name])} source rows) ===")
        out[split_name] = build_split(
            source[split_name], args.seed, cap, args.image_format, args.images_per_source
        )
        print(f"  -> {len(out[split_name])} samples")

    os.makedirs(args.output_dir, exist_ok=True)
    out.save_to_disk(args.output_dir)
    print(f"\nSaved DatasetDict -> {args.output_dir}")

    datasets_root = os.path.dirname(os.path.abspath(args.output_dir)) or "."
    import json
    if "val" in out:
        names = list(out["val"]["filename"])[: args.mini_val_size]
        with open(os.path.join(datasets_root, "mini_val_hw.json"), "w") as f:
            json.dump(names, f)
    if "test" in out:
        names = list(out["test"]["filename"])[: args.mini_test_size]
        with open(os.path.join(datasets_root, "mini_test_hw.json"), "w") as f:
            json.dump(names, f)
    print("Wrote mini_val_hw.json / mini_test_hw.json")


if __name__ == "__main__":
    main()
