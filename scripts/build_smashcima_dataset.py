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


def _load_source_dataset(source: str, slice_per_split: Optional[int] = None):
    """Load the source dataset from either a local on-disk path or the HF Hub.

    A local directory is expected to contain a ``DatasetDict`` previously
    saved with ``save_to_disk``. Anything else is treated as a Hub repo id
    and loaded with ``datasets.load_dataset`` so this script works
    unchanged on Colab where data lives on the Hub. ``token=True`` is
    passed through so gated datasets work after ``huggingface_hub.login``.

    If ``slice_per_split`` is set, only ``train[:N]``, ``val[:N]``,
    ``test[:N]`` are downloaded. This matters on Colab because PDMX-Synth
    is ~19 GB and downloading everything is wasteful when we only need a
    few hundred examples for a T4 run.
    """
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
            # Some datasets only expose train/test, etc.
            print(f"  [skip-split] {split_name}: {exc}")
    if "validation" in out and "val" not in out:
        out["val"] = out.pop("validation")
    from datasets import DatasetDict as _DD

    return _DD(out)


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


_MUSICXML_CANDIDATES = ("musicxml", "music_xml", "mxl", "xml", "score_musicxml", "musicxml_str")


def _detect_musicxml_column(source_split, requested: str) -> Optional[str]:
    """Return a usable MusicXML column name, or ``None``.

    If the user passed an explicit column name, honor it. Otherwise try a
    short list of likely names. We confirm a column is "usable" by peeking
    at the first row and checking the value is a non-empty string -- many
    HF dataset exports include a typed column with NULL everywhere.
    """
    columns = list(source_split.column_names)
    print(f"  source columns: {columns}")
    candidates = [requested] if requested and requested != "auto" else list(_MUSICXML_CANDIDATES)
    for col in candidates:
        if col in columns:
            sample = source_split[0].get(col)
            print(f"  candidate column '{col}': type={type(sample).__name__}, "
                  f"len={len(sample) if hasattr(sample, '__len__') else 'n/a'}, "
                  f"is_none={sample is None}")
            if sample is not None and (not hasattr(sample, "__len__") or len(sample) > 0):
                return col
    return None


def _iter_source_samples(source_dataset, musicxml_column: str) -> Iterable[RenderSpec]:
    skipped = 0
    yielded = 0
    for example in source_dataset:
        mx = example.get(musicxml_column)
        if mx is None or (hasattr(mx, "__len__") and len(mx) == 0):
            skipped += 1
            continue
        yielded += 1
        yield RenderSpec(
            filename=str(example.get("filename", example.get("id", f"sample_{yielded}"))),
            transcription=example.get("transcription", ""),
            musicxml=mx,
        )
    print(f"  iterated {yielded + skipped} examples (yielded={yielded}, skipped={skipped})")


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
        default="auto",
        help=(
            "Name of the column containing MusicXML strings. Use 'auto' "
            "(the default) to detect among musicxml / music_xml / mxl / "
            "xml / score_musicxml / musicxml_str."
        ),
    )
    parser.add_argument(
        "--source_split_slice",
        type=int,
        default=None,
        help=(
            "When loading from the HF Hub, only download the first N rows "
            "of each split. Saves a lot of time on Colab; the script later "
            "caps further with --max_train / --max_val / --max_test."
        ),
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
    source = _load_source_dataset(args.source_dataset, args.source_split_slice)
    print(f"Source splits: {list(source.keys())}")

    out = DatasetDict()
    for split_name, cap in (
        ("train", args.max_train),
        ("val", args.max_val),
        ("test", args.max_test),
    ):
        if split_name not in source:
            print(f"[warn] source dataset has no split '{split_name}', skipping")
            continue
        print(f"\n=== Rendering split: {split_name} ({len(source[split_name])} source rows) ===")
        col = _detect_musicxml_column(source[split_name], args.musicxml_column)
        if col is None:
            raise SystemExit(
                f"No usable MusicXML column found in split '{split_name}'.\n"
                f"  columns available: {source[split_name].column_names}\n"
                f"  PDMX-Synth ships rendered images + ABC transcriptions but does NOT\n"
                f"  contain MusicXML strings inline. Use\n"
                f"    scripts/build_handwritten_aug_dataset.py\n"
                f"  to derive a handwritten-style dataset from the existing typeset images."
            )
        print(f"  using column: {col}")
        out[split_name] = build_split(
            source[split_name], col, args.seed, cap, args.image_format
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
