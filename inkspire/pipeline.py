#pipeline.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .perusall_api import PerusallClient
from .cleaner import clean_text
from .generator import generate_annotations
from .locator import attach_text_ranges

@dataclass
class RunArgs:
    course_id: str
    document_id: str
    assignment_id: Optional[str] = None
    user_id: Optional[str] = None
    # objectives are deprecated/ignored (kept for CLI compatibility)
    objectives_path: Optional[str] = None
    kb_folder: Optional[str] = None
    provider: Optional[str] = None   # default resolved in run()
    dry_run: bool = True
    out: Optional[str] = None
    export_get_dir: Optional[str] = None  # optional: write GET artifacts if supported

def _read_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")

def _load_kb_texts(kb_folder: Optional[str]) -> List[str]:
    if not kb_folder:
        return []
    kdir = Path(kb_folder)
    if not kdir.exists():
        return []
    out: List[str] = []
    for f in sorted(kdir.iterdir()):
        if f.is_file() and f.suffix.lower() in {".txt", ".md"}:
            out.append(_read_text_file(f))
    return out

def _dbg_enabled() -> bool:
    return os.getenv("INKSPIRE_DEBUG", "0").lower() in {"1", "true"}

def _dbg(label: str, value: Any, maxlen: int = 400) -> None:
    if not _dbg_enabled():
        return
    try:
        s = value if isinstance(value, str) else repr(value)
    except Exception:
        s = "<unprintable>"
    tail = "…" if isinstance(s, str) and len(s) > maxlen else ""
    print(f"[DEBUG] {label}: {s[:maxlen]}{tail}")

def _choose_text_from_blob(blob: Any) -> Tuple[str, bool]:
    """
    Given a blob (dict/list/str), return (text, found_text_flag).
    Prefer embedded plain-text fields if present.
    """
    if isinstance(blob, dict):
        for k in ("text", "content", "plainText"):
            v = blob.get(k)
            if isinstance(v, str) and v.strip():
                return v, True
        return "", False
    if isinstance(blob, str):
        return blob, True if blob.strip() else False
    return "", False

def run(args: RunArgs) -> List[Dict]:
    """
    Flow:
      1) Fetch Perusall blob/text -> choose text (prefers blob['text']) -> clean
      2) Load KB (objectives ignored)
      3) Generate annotations (OpenAI/Gemini)
      4) Attach ranges (rangeStart, rangeEnd, fragment)
      5) Dry-run: write preview; Live: POST to Perusall and write receipts
    """
    client = PerusallClient()

    # Optional: export GET artifacts if requested and supported
    if args.export_get_dir and hasattr(client, "export_debug_artifacts"):
        try:
            paths = client.export_debug_artifacts(args.course_id, args.document_id, args.export_get_dir)
            _dbg("Exported GET artifacts", paths)
        except Exception as e:
            _dbg("export_debug_artifacts error", str(e))

    # --- 1) Fetch & clean ---
    blob_or_text: Any = None

    # Prefer rich blob if available
    if hasattr(client, "get_document_blob"):
        try:
            blob_or_text = client.get_document_blob(args.course_id, args.document_id)
            _dbg("Raw blob (repr)", blob_or_text)
        except Exception as e:
            _dbg("get_document_blob error", str(e))

    # Fallback to plain text fetch if blob is empty/unhelpful
    if (blob_or_text is None or blob_or_text == "" or blob_or_text == b"") and hasattr(client, "get_document_text"):
        try:
            blob_or_text = client.get_document_text(args.course_id, args.document_id)
            _dbg("Raw text (repr)", blob_or_text)
        except Exception as e:
            _dbg("get_document_text error", str(e))
            blob_or_text = None

    # Prefer embedded plain text if present
    preferred_text, has_preferred = _choose_text_from_blob(blob_or_text)

    # Normalizer input
    if has_preferred:
        norm_input = preferred_text
    else:
        if isinstance(blob_or_text, (dict, list)):
            try:
                norm_input = json.dumps(blob_or_text, ensure_ascii=False)
            except Exception:
                norm_input = str(blob_or_text)
        elif isinstance(blob_or_text, (bytes, bytearray)):
            norm_input = blob_or_text.decode("utf-8", errors="ignore")
        elif blob_or_text is None:
            norm_input = ""
        else:
            norm_input = str(blob_or_text)

    _dbg("Normalizer input (prefix)", norm_input[:400])

    cleaned_raw = clean_text(norm_input)
    cleaned = cleaned_raw if isinstance(cleaned_raw, str) else (str(cleaned_raw) if cleaned_raw is not None else "")
    _dbg("Cleaned text (prefix)", cleaned[:400])

    # Optional local fallback for dev
    if not cleaned.strip():
        fallback_path = Path("./content/reading_a.txt")
        if fallback_path.exists():
            cleaned = fallback_path.read_text(encoding="utf-8", errors="ignore")
            _dbg("Fallback reading_a.txt (prefix)", cleaned[:400])

    # If still empty, short-circuit
    if not cleaned.strip():
        _dbg("Abort", "No text could be fetched or cleaned. Returning empty item list.")
        out_path = Path(args.out or ("preview.json" if args.dry_run else "posted_receipts.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps([], indent=2, ensure_ascii=False), encoding="utf-8")
        return []

    # --- 2) Inputs: KB only (objectives removed) ---
    kb_texts = _load_kb_texts(args.kb_folder)

    # --- 3) Generation ---
    provider = (args.provider or os.getenv("INKSPIRE_PROVIDER") or "openai").strip().lower()
    anns: List[Dict] = generate_annotations(
        cleaned,
        kb_texts,
        provider=provider,
    )
    _dbg("Generated annotations (count)", str(len(anns)))

    # --- 4) Ranges ---
    post_payloads = attach_text_ranges(cleaned, anns)
    _dbg("Postable payloads (count)", str(len(post_payloads)))

   # --- 5) Post or dry-run ---
    if args.assignment_id and args.user_id and not args.dry_run:
        # Filter out invalid payloads up-front: Perusall will 500 on negative ranges/pages or empty text.
        valid_payloads = []
        skipped = []
        for p in post_payloads:
            ok = True
            if p.get("rangeStart", -1) < 0 or p.get("rangeEnd", -1) < 0:
                ok = False
            if p.get("rangePage", -1) < 0:
                ok = False
            txt = (p.get("text") or "").strip()
            if not txt:
                ok = False
            if ok:
                valid_payloads.append(p)
            else:
                skipped.append(p)

        _dbg("Posting: valid vs skipped", f"{len(valid_payloads)} vs {len(skipped)}")
        if skipped:
            # Write a sidecar file to inspect what was skipped
            try:
                Path("skipped_annotations.json").write_text(json.dumps(skipped, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

        if not valid_payloads:
            _dbg("Abort", "No valid annotations to post after filtering; writing empty receipts.")
            out_path = Path(args.out or "posted_receipts.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps([], indent=2, ensure_ascii=False), encoding="utf-8")
            return []

        receipts: List[Dict[str, Any]] = []

        # Optional: small retry on 5xx
        max_retries = int(os.getenv("INKSPIRE_PERUSALL_MAX_RETRIES", "2"))

        for idx, p in enumerate(valid_payloads, 1):
            attempt = 0
            while True:
                try:
                    resp = client.create_annotation(
                        course_id=args.course_id,
                        assignment_id=args.assignment_id,
                        document_id=args.document_id,
                        user_id=args.user_id,
                        fragment=p["fragment"],
                        text=p["text"],
                        range_start=p["rangeStart"],
                        range_end=p["rangeEnd"],
                        range_page=p["rangePage"],
                        range_type=p["rangeType"],
                        position_start_x=p["positionStartX"],
                        position_start_y=p["positionStartY"],
                        position_end_x=p["positionEndX"],
                        position_end_y=p["positionEndY"],
                    )
                    receipts.append(resp)
                    _dbg("Posted annotation", f"{idx}/{len(valid_payloads)} OK")
                    break
                except Exception as e:
                    attempt += 1
                    # If underlying client surfaces response text, print it
                    msg = str(e)
                    _dbg("Perusall POST error", msg)
                    if attempt > max_retries:
                        # Save the failing payload for inspection
                        try:
                            Path("failing_annotation.json").write_text(json.dumps(p, indent=2, ensure_ascii=False), encoding="utf-8")
                        except Exception:
                            pass
                        raise

        out_path = Path(args.out or "posted_receipts.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(receipts, indent=2, ensure_ascii=False), encoding="utf-8")
        return receipts


    # Dry-run: write preview
    out_path = Path(args.out or "preview.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(post_payloads, indent=2, ensure_ascii=False), encoding="utf-8")
    return post_payloads