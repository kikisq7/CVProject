"""Build a MUSCIMA++ HF dataset for out-of-distribution OMR evaluation.

MUSCIMA++ ships handwritten score images from the CVC-MUSCIMA corpus along
with symbolic annotations. For OMR-NED / TEDn evaluation we only need the
page image and a MusicXML transcription. ABC transcriptions are optional --
where unavailable, ``scripts/compute_ER.py`` will be skipped at eval time
and only the tree-/graph-based metrics are reported.

Expected on-disk layout for ``--muscima_root``::

    muscima_root/
        v2.1/
            data/
                images/*.png
                musicxml/*.xml  (optional)
                abc/*.abc       (optional)

The script is intentionally tolerant about which annotation files are
available; it will warn and include only samples that have at least an image.
"""

import argparse
import io
import json
import os
from typing import Optional

from PIL import Image
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value


def _read_text(path: Optional[str]) -> Optional[str]:
    if path is None or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _find_sibling(base_dir: str, stem: str, ext: str) -> Optional[str]:
    candidate = os.path.join(base_dir, f"{stem}{ext}")
    return candidate if os.path.isfile(candidate) else None


def build_muscima_split(
    images_dir: str,
    musicxml_dir: Optional[str],
    abc_dir: Optional[str],
    image_format: str = "PNG",
) -> Dataset:
    image_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    def generator():
        for fname in image_files:
            stem, _ = os.path.splitext(fname)
            img_path = os.path.join(images_dir, fname)
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] cannot read {img_path}: {exc}")
                continue

            buf = io.BytesIO()
            img.save(buf, format=image_format)

            musicxml = _read_text(_find_sibling(musicxml_dir, stem, ".xml")) if musicxml_dir else None
            transcription = _read_text(_find_sibling(abc_dir, stem, ".abc")) if abc_dir else None

            yield {
                "filename": stem,
                "transcription": transcription or "",
                "musicxml": musicxml or "",
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
    parser.add_argument("--images_dir", required=True, help="Directory of MUSCIMA++ page images.")
    parser.add_argument("--musicxml_dir", default=None, help="Directory of MusicXML annotations (optional).")
    parser.add_argument("--abc_dir", default=None, help="Directory of ABC annotations (optional).")
    parser.add_argument("--output_dir", default="datasets/MUSCIMA-pp")
    parser.add_argument("--split_name", default="test", choices=["train", "val", "test"])
    parser.add_argument("--image_format", default="PNG")
    args = parser.parse_args()

    ds = build_muscima_split(
        args.images_dir, args.musicxml_dir, args.abc_dir, args.image_format
    )
    dd = DatasetDict({args.split_name: ds})
    os.makedirs(args.output_dir, exist_ok=True)
    dd.save_to_disk(args.output_dir)
    print(f"Saved MUSCIMA++ -> {args.output_dir} ({len(ds)} samples in split '{args.split_name}').")

    # Drop a lightweight manifest for downstream tooling.
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump({"num_samples": len(ds), "split_name": args.split_name}, f)


if __name__ == "__main__":
    main()
