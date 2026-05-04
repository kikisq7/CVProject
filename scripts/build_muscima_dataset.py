"""Build a MUSCIMA++ HF dataset for out-of-distribution OMR evaluation.

MUSCIMA++ ships handwritten score images from the CVC-MUSCIMA corpus along
with symbolic material (MuNG graphs / optional MusicXML you supply). For
ABC-based SER/CER (what ``scripts/train.py`` and ``compute_ER.py`` expect),
set ``transcription`` either by:

* passing ``--abc_dir`` with ``.abc`` files whose stems match the page PNGs, or
* passing ``--musicxml_dir`` with ``.musicxml`` / ``.xml`` files (same stems)
  and a **MuseScore** installation so we can batch-convert MusicXML→ABC.
* using ``--auto_download`` (default) together with ``--auto_download_mung``:
  we fetch **MUSCIMA++ v2** MuNG XML. Released v2 files usually **omit**
  ``midi_pitch_code`` in ``<Data>``; we then **estimate** pitches from
  staff lines, clef, and notehead vertical position, build monophonic
  MusicXML via **music21**, then ABC via **MuseScore**.
  This reference is **approximate** (geometry + monophonic reading order).

Without any of the above, images are still usable for qualitative inference,
but token SER/CER will be undefined (empty references).

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
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

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


def _find_mung_annotations_dir(search_root: str) -> Optional[str]:
    """Return the directory under ``search_root`` with the most ``CVC-MUSCIMA_*.xml`` MuNG files."""
    best_dir, best_n = None, 0
    if not search_root or not os.path.isdir(search_root):
        return None
    for dirpath, _, filenames in os.walk(search_root):
        n = sum(
            1
            for f in filenames
            if f.startswith("CVC-MUSCIMA_") and f.lower().endswith(".xml")
        )
        if n > best_n:
            best_n = n
            best_dir = dirpath
    return best_dir


def _count_cvc_mung_xml(annotation_dir: Optional[str]) -> int:
    if not annotation_dir or not os.path.isdir(annotation_dir):
        return 0
    return sum(
        1
        for f in os.listdir(annotation_dir)
        if f.startswith("CVC-MUSCIMA_") and f.lower().endswith(".xml")
    )


def _count_cvc_mung_xml_in_tree(root: Optional[str]) -> int:
    """Count ``CVC-MUSCIMA_*.xml`` MuNG files anywhere under ``root``."""
    if not root or not os.path.isdir(root):
        return 0
    n = 0
    for _, _, files in os.walk(root):
        for f in files:
            if f.startswith("CVC-MUSCIMA_") and f.lower().endswith(".xml"):
                n += 1
    return n


def _auto_download_muscima_v2_annotations(target_dir: str) -> Optional[str]:
    """Download MUSCIMA++ v2 (MuNG XML) via omrdatasettools; return annotations directory."""
    try:
        from omrdatasettools import Downloader, OmrDataset
    except ImportError as exc:
        print(f"[muscima] omrdatasettools required for MuNG v2 download: {exc}")
        return None
    os.makedirs(target_dir, exist_ok=True)
    print(f"[muscima] downloading MuscimaPlusPlus_V2 (MuNG) into {target_dir} ...")
    try:
        Downloader().download_and_extract_dataset(OmrDataset.MuscimaPlusPlus_V2, target_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[muscima] MuNG v2 download/extract failed: {exc}")
        return None
    found = _find_mung_annotations_dir(target_dir)
    if found:
        print(f"[muscima] MuNG XML dir: {found} ({_count_cvc_mung_xml(found)} files)")
    else:
        print("[muscima] warning: could not locate MuNG XML after v2 extract.")
    return found


_MUNG_IMPORT_WARNED = False
_MUNG_GEOM_WARNED = False

# Nine staff positions from top staff line downward (treble / bass). Release MuNG
# usually omits ``midi_pitch_code``; these tables support geometry-based estimates.
_TREBLE_MIDIS_TOP_TO_BOTTOM: Tuple[int, ...] = (77, 76, 74, 72, 71, 69, 67, 65, 64)
_BASS_MIDIS_TOP_TO_BOTTOM: Tuple[int, ...] = (57, 55, 53, 52, 50, 48, 47, 45, 43)


def _ycenter(n) -> float:
    return float(n.top) + float(n.height) * 0.5


def _find_mung_xml_path(search_root: Optional[str], stem: str) -> Optional[str]:
    """Locate ``CVC-MUSCIMA_*.xml`` for page ``stem`` under ``search_root`` (recursive)."""
    if not search_root or not os.path.isdir(search_root):
        return None
    want = [f"{stem}.xml", f"CVC-MUSCIMA_{stem}.xml"]
    for name in want:
        direct = os.path.join(search_root, name)
        if os.path.isfile(direct):
            return direct
    if stem.startswith("CVC-MUSCIMA_"):
        short = stem[len("CVC-MUSCIMA_") :]
        alt = os.path.join(search_root, f"{short}.xml")
        if os.path.isfile(alt):
            return alt
    stem_l = stem.lower()
    for dirpath, _, files in os.walk(search_root):
        for f in files:
            if not f.lower().endswith(".xml") or not f.startswith("CVC-MUSCIMA_"):
                continue
            base = f[:-4]
            if base == stem or base.lower() == stem_l:
                return os.path.join(dirpath, f)
    return None


def _group_stafflines_by_staff(
    staff_boxes: List, staff_lines: List
) -> Dict[int, List]:
    groups: Dict[int, List] = {}
    for st in staff_boxes:
        groups[id(st)] = []
        y0, y1 = float(st.top), float(st.top + st.height)
        for ln in staff_lines:
            yc = _ycenter(ln)
            if y0 - 3.0 <= yc <= y1 + 3.0:
                groups[id(st)].append(ln)
    return groups


def _clef_kind_for_staff(staff, clefs: Sequence) -> str:
    y0, y1 = float(staff.top), float(staff.top + staff.height)
    sx = float(staff.left)
    for c in clefs:
        cy = _ycenter(c)
        if not (y0 - 15.0 <= cy <= y1 + 15.0):
            continue
        if float(c.left) > sx + 1200.0:
            continue
        name = (c.class_name or "")
        if name == "gClef":
            return "treble"
        if name == "fClef":
            return "bass"
    return "treble"


def _midis_from_staff_geometry(nodes) -> List[int]:
    """Estimate MIDI pitches from staff lines + clef + notehead Y (release MuNG has no Data pitch)."""
    global _MUNG_GEOM_WARNED
    staff_boxes = [n for n in nodes if (n.class_name or "") == "staff"]
    staff_lines = [n for n in nodes if (n.class_name or "") == "staffLine"]
    clefs = [n for n in nodes if (n.class_name or "") in ("gClef", "fClef")]
    noteheads = [
        n
        for n in nodes
        if "notehead" in (n.class_name or "").lower()
        and "rest" not in (n.class_name or "").lower()
        and "grace" not in (n.class_name or "").lower()
    ]
    if not staff_boxes or not staff_lines or not noteheads:
        if not _MUNG_GEOM_WARNED:
            print(
                "[muscima] MuNG geometry: need staff + staffLine + notehead nodes "
                f"(have {len(staff_boxes)}/{len(staff_lines)}/{len(noteheads)})."
            )
            _MUNG_GEOM_WARNED = True
        return []
    groups = _group_stafflines_by_staff(staff_boxes, staff_lines)
    # Order systems: top-to-bottom, then left-to-right.
    sorted_staffs = sorted(
        staff_boxes,
        key=lambda s: (float(s.top) + float(s.height) * 0.5, float(s.left)),
    )
    events: List[Tuple[float, float, int]] = []
    for st in sorted_staffs:
        lines = groups.get(id(st), [])
        if len(lines) < 3:
            continue
        line_ys = sorted(_ycenter(ln) for ln in lines)
        y_top, y_bot = line_ys[0], line_ys[-1]
        if y_bot <= y_top:
            continue
        table = _TREBLE_MIDIS_TOP_TO_BOTTOM
        if _clef_kind_for_staff(st, clefs) == "bass":
            table = _BASS_MIDIS_TOP_TO_BOTTOM
        sy0, sy1 = float(st.top), float(st.top + st.height)
        for nh in noteheads:
            yc = _ycenter(nh)
            if not (sy0 - 5.0 <= yc <= sy1 + 5.0):
                continue
            frac = (yc - y_top) / (y_bot - y_top) * 8.0
            idx = int(round(frac))
            idx = max(0, min(8, idx))
            midi_i = table[idx]
            events.append((float(st.top), float(nh.left), midi_i))
    if not events:
        if not _MUNG_GEOM_WARNED:
            print("[muscima] MuNG geometry: no noteheads matched staff boxes.")
            _MUNG_GEOM_WARNED = True
        return []
    events.sort(key=lambda t: (t[0], t[1]))
    if len(events) > 10000:
        events = events[:10000]
    return [m for _, _, m in events]


def _extract_midis_from_mung_xml(path: str) -> List[int]:
    global _MUNG_IMPORT_WARNED
    try:
        from mung.io import read_nodes_from_file
    except ImportError:
        if not _MUNG_IMPORT_WARNED:
            print("[muscima] pip install mung lxml to parse MuNG annotations.")
            _MUNG_IMPORT_WARNED = True
        return []
    try:
        nodes = read_nodes_from_file(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[muscima] read_nodes_from_file failed ({path}): {exc}")
        return []
    if not nodes:
        return []
    events = []
    for n in nodes:
        cls = (n.class_name or "").lower()
        if "notehead" not in cls:
            continue
        data = getattr(n, "data", None)
        if not data:
            continue
        midi = data.get("midi_pitch_code")
        if midi is None:
            continue
        try:
            midi_i = int(midi)
        except (TypeError, ValueError):
            continue
        events.append((n.top, n.left, midi_i))
    if events:
        staff_band = 80
        events.sort(key=lambda t: (-(t[0] // staff_band), t[1]))
        return [m for _, _, m in events]

    return _midis_from_staff_geometry(nodes)


def _midis_to_abc_primitive(midis: List[int]) -> str:
    """Last-resort monophonic ABC (quarter notes) without MuseScore."""
    if not midis:
        return ""
    try:
        from music21.pitch import Pitch
    except ImportError:
        return ""
    tokens = []
    for m in midis:
        p = Pitch(midi=int(m))
        acc = ""
        if p.accidental is not None and p.accidental.alter != 0:
            al = float(p.accidental.alter)
            if al == 1.0:
                acc = "^"
            elif al == -1.0:
                acc = "_"
            elif al == 2.0:
                acc = "^^"
            elif al == -2.0:
                acc = "__"
        step = p.step
        oct_shift = int(p.octave) - 4
        if oct_shift >= 0:
            body = step.lower() + acc + ("'" * oct_shift)
        else:
            body = step.upper() + acc + ("," * (-oct_shift))
        tokens.append(body)
    voice = " ".join(tokens)
    return (
        "X:1\n"
        "T:muscima-ref\n"
        "M:4/4\n"
        "L:1/4\n"
        "K:none\n"
        f"| {voice} |\n"
    )


def _midis_to_abc_via_music21_musescore(
    midis: List[int],
    *,
    musescore_bin: Optional[str] = None,
) -> str:
    """Monophonic quarter-note stream → MusicXML (music21) → ABC (MuseScore)."""
    if not midis:
        return ""
    try:
        from music21 import note as m21note
        from music21 import stream as m21stream
    except ImportError:
        print("[muscima] pip install music21 for MuNG→MusicXML step.")
        return ""
    s = m21stream.Stream()
    for m in midis:
        s.append(m21note.Note(midi=int(m)))
    try:
        with tempfile.TemporaryDirectory() as td:
            xmlp = os.path.join(td, "ref.musicxml")
            s.write("musicxml", fp=xmlp)
            mx = _read_text(xmlp)
            if not mx:
                return _midis_to_abc_primitive(midis)
            out = _musicxml_to_abc(mx, musescore_bin=musescore_bin)
            if out.strip():
                return out
    except Exception as exc:  # noqa: BLE001
        print(f"[muscima] music21 MusicXML export failed: {exc}")
    return _midis_to_abc_primitive(midis)


def _mung_xml_to_abc(path: str, *, musescore_bin: Optional[str] = None) -> str:
    midis = _extract_midis_from_mung_xml(path)
    return _midis_to_abc_via_music21_musescore(midis, musescore_bin=musescore_bin)


def build_muscima_split(
    images_dir: str,
    musicxml_dir: Optional[str],
    abc_dir: Optional[str],
    image_format: str = "PNG",
    *,
    convert_musicxml_to_abc: bool = True,
    musescore_bin: Optional[str] = None,
    mung_annotations_dir: Optional[str] = None,
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

            mung_path = (
                _find_mung_xml_path(mung_annotations_dir, stem)
                if mung_annotations_dir
                else None
            )
            if not transcription and mung_path:
                transcription = _mung_xml_to_abc(
                    mung_path, musescore_bin=musescore_bin
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
    if n_tx and mung_annotations_dir:
        print(
            "[muscima] Note: references from MuNG are monophonic approximations "
            "(for SER/CER); not identical to full scores."
        )
    if not n_tx and mung_annotations_dir and convert_musicxml_to_abc:
        print(
            "[muscima] No ABC from MuNG/convert path. "
            "Install: `pip install mung music21 lxml`, `apt install musescore3`, "
            "or pass `--abc_dir` / `--musicxml_dir`. "
            "If MuNG downloaded but MIDI list empty, check MuNG parse logs above."
        )
    if not n_tx and not mung_annotations_dir and n_mx and convert_musicxml_to_abc:
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
    parser.add_argument(
        "--mung_annotations_dir",
        default=None,
        help="Directory with MUSCIMA++ v2 MuNG XML (CVC-MUSCIMA_*.xml). Overrides auto-download.",
    )
    parser.add_argument(
        "--auto_download_mung",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download/extract MUSCIMA++ v2 MuNG next to images under download_dir/muscima_v2 (default: True).",
    )
    args = parser.parse_args()

    images_dir = args.images_dir
    if args.auto_download or images_dir is None:
        images_dir = _auto_download_muscima_images(args.download_dir)

    mung_search_root: Optional[str] = None
    if args.mung_annotations_dir:
        mung_search_root = os.path.abspath(args.mung_annotations_dir)
    elif args.auto_download_mung:
        v2_root = os.path.join(args.download_dir, "muscima_v2")
        if _count_cvc_mung_xml_in_tree(v2_root) < 130:
            _auto_download_muscima_v2_annotations(v2_root)
        mung_search_root = v2_root

    ds = build_muscima_split(
        images_dir,
        args.musicxml_dir,
        args.abc_dir,
        args.image_format,
        convert_musicxml_to_abc=args.convert_musicxml_to_abc,
        musescore_bin=args.musescore_bin,
        mung_annotations_dir=mung_search_root,
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
            "mung_annotations_dir": mung_search_root,
            "mung_cvc_xml_count": _count_cvc_mung_xml_in_tree(mung_search_root),
            "auto_download_mung": args.auto_download_mung,
            "has_abc_dir": bool(args.abc_dir),
            "has_musicxml_dir": bool(args.musicxml_dir),
            "convert_musicxml_to_abc": args.convert_musicxml_to_abc,
            "non_empty_transcription_count": sum(1 for t in ds["transcription"] if t),
            "non_empty_musicxml_count": sum(1 for t in ds["musicxml"] if t),
        }, f)


if __name__ == "__main__":
    main()
