"""Build the PDMX-SmashcimaHW synthetic handwritten dataset.

This script re-renders the MusicXML sources used by PDMX-Synth through
Smashcima [1] to produce handwritten-style page images, and pairs each
rendered image with the corresponding ABC transcription already available
in PDMX-Synth. Because labels are inherited from PDMX-Synth rather than
re-derived from the Smashcima output, image-label alignment is guaranteed.

The output is a Hugging Face ``DatasetDict`` with ``train`` / ``val`` /
``test`` splits, each containing ``image``, ``transcription``, ``filename``
and (optionally) ``musicxml`` columns. The schema matches what
``scripts/train.py`` expects via ``load_from_disk``.

References
----------
[1] Fiser et al., "Smashcima: a framework that produces handwritten-looking
    score images", 2024.
"""

import argparse
import io
import json
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional

from PIL import Image
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


def _load_source_dataset(source: str):
    """Load the source dataset from either a local on-disk path or the HF Hub.

    A local directory is expected to contain a ``DatasetDict`` previously
    saved with ``save_to_disk``. Anything else is treated as a Hub repo id
    and loaded with ``datasets.load_dataset`` so this script works
    unchanged on Colab where data lives on the Hub.
    """
    if os.path.isdir(source):
        return load_from_disk(source)
    return load_dataset(source)


@dataclass
class RenderSpec:
    filename: str
    transcription: str
    musicxml: Optional[str]


def _render_one_page(musicxml_str: str, seed: int) -> Image.Image:
    """Render a MusicXML string to a PIL handwritten-style image via Smashcima.

    Smashcima's public API is still evolving; we keep the call site narrow so
    that callers can override the rendering backend if needed.
    """
    import smashcima  # lazy import; only needed at dataset-build time

    # Smashcima exposes a high-level renderer that consumes MusicXML and
    # returns a raster image. See https://github.com/OMR-Research/smashcima
    # for the current API surface.
    renderer = smashcima.BasicSynthesizer(seed=seed)
    page_images: List[Image.Image] = renderer.synthesize_from_musicxml(musicxml_str)
    if not page_images:
        raise RuntimeError("Smashcima returned no pages")
    return page_images[0]


def _iter_source_samples(source_dataset, musicxml_column: str) -> Iterable[RenderSpec]:
    for example in source_dataset:
        mx = example.get(musicxml_column)
        if mx is None:
            continue
        yield RenderSpec(
            filename=str(example.get("filename", example.get("id", "unknown"))),
            transcription=example["transcription"],
            musicxml=mx,
        )


def build_split(
    source_split,
    musicxml_column: str,
    seed: int,
    max_samples: Optional[int],
    image_format: str,
) -> Dataset:
    specs = list(_iter_source_samples(source_split, musicxml_column))
    if max_samples is not None:
        specs = specs[:max_samples]

    rng = random.Random(seed)

    def generator():
        for idx, spec in enumerate(tqdm(specs, desc="Rendering")):
            try:
                img = _render_one_page(spec.musicxml, seed=rng.randint(0, 2**31 - 1))
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {spec.filename}: {exc}")
                continue
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format=image_format)
            yield {
                "filename": spec.filename,
                "transcription": spec.transcription,
                "musicxml": spec.musicxml,
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
    parser.add_argument(
        "--source_dataset",
        default="datasets/PDMX-Synth",
        help=(
            "Source dataset with MusicXML + ABC transcription. "
            "May be either a local directory previously saved via "
            "Dataset.save_to_disk (e.g. datasets/PDMX-Synth) or a HF Hub "
            "repo id such as guangyangmusic/PDMX-Synth (handy on Colab)."
        ),
    )
    parser.add_argument(
        "--musicxml_column",
        default="musicxml",
        help="Name of the column containing MusicXML strings.",
    )
    parser.add_argument(
        "--output_dir",
        default="datasets/PDMX-SmashcimaHW",
        help="Where to save the resulting DatasetDict.",
    )
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_val", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_format", default="PNG")
    parser.add_argument(
        "--mini_val_size", type=int, default=200,
        help="Size of the companion datasets/mini_val_hw.json quick-eval subset.",
    )
    parser.add_argument(
        "--mini_test_size", type=int, default=200,
        help="Size of the companion datasets/mini_test_hw.json quick-eval subset.",
    )
    args = parser.parse_args()

    print(f"Loading source dataset from {args.source_dataset}")
    source = _load_source_dataset(args.source_dataset)

    out = DatasetDict()
    for split_name, cap in (
        ("train", args.max_train),
        ("val", args.max_val),
        ("test", args.max_test),
    ):
        if split_name not in source:
            print(f"[warn] source dataset has no split '{split_name}', skipping")
            continue
        print(f"Rendering split: {split_name}")
        out[split_name] = build_split(
            source[split_name], args.musicxml_column, args.seed, cap, args.image_format
        )

    os.makedirs(args.output_dir, exist_ok=True)
    out.save_to_disk(args.output_dir)
    print(f"Saved DatasetDict -> {args.output_dir}")

    # Build mini-eval filename subsets so configs/legato-dlora.json can
    # point at them for fast iteration during training.
    datasets_root = os.path.dirname(os.path.abspath(args.output_dir)) or "."
    if "val" in out:
        names = list(out["val"]["filename"])[: args.mini_val_size]
        with open(os.path.join(datasets_root, "mini_val_hw.json"), "w") as f:
            json.dump(names, f)
    if "test" in out:
        names = list(out["test"]["filename"])[: args.mini_test_size]
        with open(os.path.join(datasets_root, "mini_test_hw.json"), "w") as f:
            json.dump(names, f)


if __name__ == "__main__":
    main()
