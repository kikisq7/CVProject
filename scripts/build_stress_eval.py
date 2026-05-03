"""Generate augmented "stress test" eval datasets from a base test split.

Idea: when only a single GPU-hour is available for evaluation, we can still
demonstrate model robustness by evaluating against *multiple visual
perturbations* of the same base ground truth. Each perturbation becomes its
own HF dataset under ``--out_root/<perturbation>``, and the unified evaluation
loop in ``scripts/eval_legato_t4.sh`` iterates over them.

Perturbations (all label-preserving):

* ``clean``     -- copy of the base split, no augmentation.
* ``rotate``    -- small random rotation (+/- 4 deg) to simulate page skew.
* ``blur``      -- Gaussian blur, simulating low-quality scans.
* ``noise``     -- additive Gaussian noise + slight contrast jitter.
* ``jpeg``      -- aggressive JPEG re-compression (quality 30) to simulate
                   web/photographed scores.
* ``downscale`` -- 0.5x resize then upscale, simulating low-DPI capture.

The base split is selected by ``--split`` (default ``test``). The script
writes a ``DatasetDict({split_name: ...})`` for each perturbation so it
plugs straight into ``scripts/train.py do_predict=true``.
"""

import argparse
import io
import os
import random
from typing import Callable, Dict

from PIL import Image, ImageFilter
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value, load_from_disk
from tqdm import tqdm


def _to_pil(example_image) -> Image.Image:
    if isinstance(example_image, Image.Image):
        return example_image.convert("RGB")
    return Image.open(example_image).convert("RGB")


def _aug_clean(img: Image.Image, rng: random.Random) -> Image.Image:
    return img


def _aug_rotate(img: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-4.0, 4.0)
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255), expand=False)


def _aug_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 1.6)))


def _aug_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    import numpy as np

    arr = np.asarray(img).astype("int16")
    sigma = rng.uniform(6, 14)
    noise = np.random.normal(0, sigma, size=arr.shape).astype("int16")
    arr = (arr + noise).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def _aug_jpeg(img: Image.Image, rng: random.Random) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=rng.choice([25, 30, 40]))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _aug_downscale(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    factor = rng.uniform(0.45, 0.6)
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


PERTURBATIONS: Dict[str, Callable[[Image.Image, random.Random], Image.Image]] = {
    "clean": _aug_clean,
    "rotate": _aug_rotate,
    "blur": _aug_blur,
    "noise": _aug_noise,
    "jpeg": _aug_jpeg,
    "downscale": _aug_downscale,
}


def _save_perturbed(
    base_split,
    perturb_name: str,
    transform: Callable[[Image.Image, random.Random], Image.Image],
    out_dir: str,
    split_name: str,
    seed: int,
    image_format: str,
) -> Dataset:
    rng = random.Random(seed)

    def generator():
        for example in tqdm(base_split, desc=perturb_name):
            try:
                img = _to_pil(example["image"])
                img = transform(img, rng)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {example.get('filename')}: {exc}")
                continue
            buf = io.BytesIO()
            img.save(buf, format=image_format)
            yield {
                "filename": example.get("filename", ""),
                "transcription": example.get("transcription", ""),
                "musicxml": example.get("musicxml", ""),
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
    ds = Dataset.from_generator(generator, features=features)
    DatasetDict({split_name: ds}).save_to_disk(out_dir)
    return ds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_dataset", default="datasets/PDMX-SmashcimaHW",
        help="Local HF dataset directory (must contain the chosen split).",
    )
    parser.add_argument(
        "--split", default="test",
        help="Which split of the source dataset to perturb.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=200,
        help="Cap per output dataset; T4 evaluation budget.",
    )
    parser.add_argument(
        "--out_root", default="datasets/stress",
        help="Output directory; one subdirectory per perturbation.",
    )
    parser.add_argument(
        "--perturbations", nargs="+", default=list(PERTURBATIONS.keys()),
        choices=list(PERTURBATIONS.keys()),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_format", default="PNG")
    args = parser.parse_args()

    print(f"Loading source dataset from {args.source_dataset}")
    src = load_from_disk(args.source_dataset)
    if args.split not in src:
        raise SystemExit(f"split '{args.split}' not in source DatasetDict")
    base = src[args.split].select(range(min(args.max_samples, len(src[args.split]))))
    print(f"Base split has {len(base)} samples after capping.")

    os.makedirs(args.out_root, exist_ok=True)
    for name in args.perturbations:
        out_dir = os.path.join(args.out_root, name)
        if os.path.isdir(out_dir):
            print(f"[skip] {out_dir} already exists; delete it to rebuild.")
            continue
        ds = _save_perturbed(
            base,
            name,
            PERTURBATIONS[name],
            out_dir,
            args.split,
            args.seed,
            args.image_format,
        )
        print(f"  -> {out_dir} ({len(ds)} samples)")


if __name__ == "__main__":
    main()
