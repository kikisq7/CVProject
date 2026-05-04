"""Build a MUSCIMA++ HF dataset for out-of-distribution OMR evaluation.

MUSCIMA++ ships handwritten score images from the CVC-MUSCIMA corpus along
with symbolic material (MuNG graphs / optional MusicXML you supply). For
ABC-based SER/CER (what ``scripts/train.py`` and ``compute_ER.py`` expect),
set ``transcription`` either by:

* passing ``--abc_dir`` with ``.abc`` files whose stems match the page PNGs, or
* passing ``--musicxml_dir`` with ``.musicxml`` / ``.xml`` files (same stems)
  and a **MuseScore** installation so we can batch-convert MusicXML→ABC.

Without ABC or convertible MusicXML, images are still usable for qualitative
inference, but token SER/CER will be undefined (empty references).

Expected on-disk layout (when you supply symbols)::

    muscima_root/
        images/*.png
        musicxml/*.xml   (optional; converted to ABC if MuseScore is available)
        abc/*.abc        (optional; overrides MusicXML-derived ABC)

The script is tolerant: every sample must have an image; symbols are optional.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import tempfile
from typing import Iterator, Optional

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


def _find_musicxml_path(musicxml_dir: str, stem: str) -> Optional[str]:
    for ext in (".musicxml", ".xml"):
        p = _find_sibling(musicxml_dir, stem, ext)
        if p:
            return p
    return None


def _musescore_candidates(explicit: Optional[str] = None) -> Iterator[str]:
    if explicit:
        yield explicit
    env_bin = os.environ.get("MUSESCORE_EXECUTABLE")
    if env_bin:
        yield env_bin
    for name in ("musescore3", "mscore3", "musescore", "mscore"):
        found = shutil.which(name)
        if found:
            yield found
    here = os.path.dirname(os.path.abspath(__file__))
    legacy = os.path.normpath(os.path.join(here, "..", "software", "mscore"))
    if os.path.isfile(legacy) or os.path.islink(legacy):
        yield legacy


_MUSESCORE_CONVERT_WARNED = False


def _musicxml_to_abc(
    musicxml: str,
    *,
    musescore_bin: Optional[str] = None,
    timeout_s: int = 180,
) -> str:
    """Convert a MusicXML document string to ABC via MuseScore batch mode."""
    global _MUSESCORE_CONVERT_WARNED
    if not musicxml or not musicxml.strip():
        return ""

    env = os.environ.copy()
    # Headless Linux / Colab: avoid Qt display errors.
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    last_err = ""
    for mscore in _musescore_candidates(musescore_bin):
        if not mscore:
            continue
        try:
            with tempfile.TemporaryDirectory() as td:
                xml_path = os.path.join(td, "score.musicxml")
                abc_path = os.path.join(td, "score.abc")
                with open(xml_path, "w", encoding="utf-8") as f:
                    f.write(musicxml)
                proc = subprocess.run(
                    [mscore, xml_path, "-o", abc_path, "-f"],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    env=env,
                    check=False,
                )
                if proc.returncode != 0:
                    last_err = (proc.stderr or proc.stdout or "")[:500]
                    continue
                if os.path.isfile(abc_path):
                    with open(abc_path, encoding="utf-8", errors="ignore") as f:
                        return f.read()
                last_err = "MuseScore finished but score.abc missing"
        except (subprocess.SubprocessError, OSError) as exc:
            last_err = str(exc)
            continue
    if last_err and not _MUSESCORE_CONVERT_WARNED:
        print(
            "[muscima] MusicXML→ABC via MuseScore failed "
            f"({last_err[:200]!r}). "
            "Install MuseScore (e.g. `apt install musescore3` on Debian/Ubuntu) "
            "or set MUSESCORE_EXECUTABLE."
        )
        _MUSESCORE_CONVERT_WARNED = True
    return ""


def build_muscima_split(
    images_dir: str,
    musicxml_dir: Optional[str],
    abc_dir: Optional[str],
    image_format: str = "PNG",
    *,
    convert_musicxml_to_abc: bool = True,
    musescore_bin: Optional[str] = None,
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

            mx_path = _find_musicxml_path(musicxml_dir, stem) if musicxml_dir else None
            musicxml = _read_text(mx_path) if mx_path else None
            transcription = _read_text(_find_sibling(abc_dir, stem, ".abc")) if abc_dir else None

            if not transcription and musicxml and convert_musicxml_to_abc:
                transcription = _musicxml_to_abc(
                    musicxml, musescore_bin=musescore_bin
                )

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
    ds = Dataset.from_generator(generator, features=features)
    n_tx = sum(1 for t in ds["transcription"] if t)
    n_mx = sum(1 for t in ds["musicxml"] if t)
    print(
        f"[muscima] Built {len(ds)} pages: "
        f"{n_tx} with non-empty ABC `transcription`, {n_mx} with raw `musicxml` text."
    )
    if n_mx and not n_tx and convert_musicxml_to_abc:
        print(
            "[muscima] No ABC transcriptions were produced. "
            "Install MuseScore (`apt install musescore3` on Colab/Ubuntu) "
            "or pass matching `.abc` files via --abc_dir."
        )
    return ds


def _auto_download_muscima_images(target_dir: str) -> str:
    """Download the 140 CVC-MUSCIMA pages used by MUSCIMA++ via omrdatasettools.

    Returns the directory containing the .png page images. Tries multiple
    layouts because the zip's internal directory structure has changed
    across versions of the OMR-Datasets release.
    """
    try:
        from omrdatasettools import Downloader, OmrDataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "omrdatasettools is required for --auto_download. "
            "pip install omrdatasettools"
        ) from exc

    os.makedirs(target_dir, exist_ok=True)
    print(f"[muscima] downloading MuscimaPlusPlus_Images into {target_dir} ...")
    Downloader().download_and_extract_dataset(OmrDataset.MuscimaPlusPlus_Images, target_dir)

    # Walk and find the directory with the most .png files.
    best_dir, best_count = target_dir, 0
    for root, _, files in os.walk(target_dir):
        png_count = sum(1 for f in files if f.lower().endswith(".png"))
        if png_count > best_count:
            best_dir, best_count = root, png_count
    print(f"[muscima] auto-detected images dir: {best_dir} ({best_count} PNGs)")
    if best_count == 0:
        raise SystemExit("MUSCIMA++ images download did not yield any PNGs.")
    return best_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images_dir", default=None, help="Directory of MUSCIMA++ page images. Omit when using --auto_download.")
    parser.add_argument("--musicxml_dir", default=None, help="Directory of MusicXML annotations (optional).")
    parser.add_argument("--abc_dir", default=None, help="Directory of ABC annotations (optional).")
    parser.add_argument("--output_dir", default="datasets/MUSCIMA-pp")
    parser.add_argument("--split_name", default="test", choices=["train", "val", "test"])
    parser.add_argument("--image_format", default="PNG")
    parser.add_argument(
        "--auto_download", action="store_true",
        help="Download the 140 MUSCIMA++ page images via omrdatasettools (Colab-friendly).",
    )
    parser.add_argument(
        "--download_dir", default="downloads/muscima_pp",
        help="Where to extract the auto-downloaded MUSCIMA++ images.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Optional cap on the number of pages to include (handy for fast Colab eval).",
    )
    parser.add_argument(
        "--convert_musicxml_to_abc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When MusicXML is present and no .abc exists, run MuseScore to fill transcription (default: on).",
    )
    parser.add_argument(
        "--musescore_bin",
        default=None,
        help="Path to MuseScore executable (overrides auto-detect / MUSESCORE_EXECUTABLE).",
    )
    args = parser.parse_args()

    images_dir = args.images_dir
    if args.auto_download or images_dir is None:
        images_dir = _auto_download_muscima_images(args.download_dir)

    ds = build_muscima_split(
        images_dir,
        args.musicxml_dir,
        args.abc_dir,
        args.image_format,
        convert_musicxml_to_abc=args.convert_musicxml_to_abc,
        musescore_bin=args.musescore_bin,
    )
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    dd = DatasetDict({args.split_name: ds})
    os.makedirs(args.output_dir, exist_ok=True)
    dd.save_to_disk(args.output_dir)
    print(f"Saved MUSCIMA++ -> {args.output_dir} ({len(ds)} samples in split '{args.split_name}').")

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump({
            "num_samples": len(ds),
            "split_name": args.split_name,
            "source_images_dir": images_dir,
            "has_abc_dir": bool(args.abc_dir),
            "has_musicxml_dir": bool(args.musicxml_dir),
            "convert_musicxml_to_abc": args.convert_musicxml_to_abc,
            "non_empty_transcription_count": sum(1 for t in ds["transcription"] if t),
            "non_empty_musicxml_count": sum(1 for t in ds["musicxml"] if t),
        }, f)


if __name__ == "__main__":
    main()
