from __future__ import annotations

import os
import re
import json
from typing import Optional, Dict, Any, Union, List

import httpx
from .config import settings  # <- your Settings loader

DEFAULT_BASE_URL = "https://app.perusall.com/api/v1"


# ----------------------------- helpers ----------------------------- #

def _normalize_text(s: str) -> str:
    """
    Pragmatic PDF-text normalizer:
      - join hyphenated line breaks: 'inter-\n national' -> 'international'
      - tidy linebreak spacing
      - compress >2 blank lines to exactly two
      - collapse runs of spaces/tabs
      - trim spaces before punctuation
    """
    if not s:
        return ""
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)      # dehyphenate across line breaks
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)            # tidy lone linebreaks
    s = re.sub(r"\n{3,}", "\n\n", s)                  # compress blank lines
    s = re.sub(r"[ \t]{2,}", " ", s)                  # collapse spaces/tabs
    s = re.sub(r"\s+([,:;.!?])", r"\1", s)            # remove space before punctuation
    return s.strip()


def _parse_pdfjs_items(items: List[Dict[str, Any]]) -> str:
    """
    Reconstruct text from a PDF.js-like 'items' array containing {'str': '...', 'hasEOL': bool}.
    """
    parts: List[str] = []
    line: List[str] = []
    for it in items:
        s = it.get("str")
        if isinstance(s, str):
            line.append(s)
        if it.get("hasEOL") is True:
            parts.append(" ".join(line).strip())
            line = []
    if line:
        parts.append(" ".join(line).strip())
    return "\n".join(p for p in parts if p)


def _parse_text_json(obj: Any) -> str:
    """
    Best-effort extraction from common shapes:
      - {'items': [...] }            -> PDF.js text layer
      - {'blocks': [{'text': ...}] } -> block text array
      - {'pages': [{'items': [...]}]} (nested) -> collect items
      - fallback: collect string leaves with a budget cap
    """
    # 1) PDF.js 'items'
    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        return _parse_pdfjs_items(obj["items"])

    # 2) Blocks with 'text'
    if isinstance(obj, dict) and isinstance(obj.get("blocks"), list):
        texts: List[str] = []
        for b in obj["blocks"]:
            t = b.get("text")
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
        if texts:
            return "\n".join(texts)

    # 3) Nested pages[i].items
    if isinstance(obj, dict) and isinstance(obj.get("pages"), list):
        out: List[str] = []
        for p in obj["pages"]:
            it = p.get("items")
            if isinstance(it, list):
                out.append(_parse_pdfjs_items(it))
        if any(out):
            return "\n\n".join(o for o in out if o.strip())

    # 4) Generic fallback: collect string leaves (bounded)
    collected: List[str] = []
    budget = 0

    def _walk(x: Any) -> None:
        nonlocal budget
        if budget > 5000:  # cap to avoid giant dumps
            return
        if isinstance(x, str):
            s = x.strip()
            if s:
                collected.append(s)
                budget += len(s)
        elif isinstance(x, list):
            for y in x:
                _walk(y)
        elif isinstance(x, dict):
            for y in x.values():
                _walk(y)

    _walk(obj)
    return " ".join(collected) if collected else ""


# ----------------------------- client ----------------------------- #

class PerusallClient:
    """
    Minimal Perusall API client.
    Reads:
      - PERUSALL_BASE_URL (optional, default https://app.perusall.com/api/v1)
      - PERUSALL_INSTITUTION (required)
      - PERUSALL_API_TOKEN (required)
    """

    def __init__(self, base_url: Optional[str] = None):
        base = (
            base_url
            or settings.perusall_base_url
            or os.getenv("PERUSALL_BASE_URL")
            or DEFAULT_BASE_URL
        )
        self.base_url = base.rstrip("/")

        inst = settings.perusall_institution
        token = settings.perusall_api_token
        if not inst or not token:
            raise RuntimeError(
                "Missing PERUSALL_INSTITUTION or PERUSALL_API_TOKEN in env/.env. "
                "Set both to authenticate to Perusall."
            )

        # follow_redirects is useful for expiring/redirecting textContentUrls
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={
                "X-Institution": inst,
                "X-API-Token": token,
                "Accept": "application/json, text/plain;q=0.9",
            },
            timeout=30.0,
            follow_redirects=True,
        )

    # ----------- READ ----------- #

    def _fetch_page_payloads(self, pages: List[Dict[str, Any]]) -> List[str]:
        """
        Download each page's raw payload from its textContentUrl.
        Returns a list of *raw strings* (JSON text or plain text), preserving order by page number.
        """
        raw_payloads: List[str] = []
        for p in sorted(pages, key=lambda x: x.get("number", 0)):
            url = p.get("textContentUrl")
            if not url:
                continue
            try:
                r = self.client.get(url)
            except httpx.HTTPError:
                continue

            if r.status_code != 200:
                continue

            raw = r.text or ""
            if raw.strip():
                raw_payloads.append(raw.strip())
        return raw_payloads

    def _extract_text_from_payloads(self, raw_payloads: List[str]) -> str:
        """
        Turn raw page payloads (JSON or plain text) into a single plain-text document.
        """
        page_texts: List[str] = []
        for raw in raw_payloads:
            text = ""
            # Detect JSON
            if raw.lstrip().startswith(("{", "[")):
                try:
                    obj = json.loads(raw)
                    text = _parse_text_json(obj)
                except Exception:
                    text = ""  # fall through to plain-text fallback below
            # Fallback: treat as plain text if JSON parse yielded nothing
            if not text and raw.strip():
                text = raw
            if text.strip():
                page_texts.append(text.strip())

        combined = "\n\n".join(page_texts)
        return _normalize_text(combined)

    def get_document_blob(self, course_id: str, document_id: str) -> Union[Dict[str, Any], str]:
        """
        Fetch the library item metadata + page descriptors for a document,
        and embed a materialized 'text' field with the full cleaned content.
        """
        url = f"/courses/{course_id}/library/{document_id}"
        try:
            r = self.client.get(url)
        except httpx.HTTPError:
            return ""

        if r.status_code == 403:
            # Common when item is catalog-protected (publisher restriction)
            return {"_error": "forbidden_or_catalog", "_status": 403}
        if r.status_code == 404:
            return {"_error": "not_found", "_status": 404}
        if r.status_code != 200:
            return ""

        try:
            blob = r.json()
        except Exception:
            return ""

        # If pages[] present, download and attach materialized text
        if isinstance(blob, dict) and isinstance(blob.get("pages"), list):
            page_raws = self._fetch_page_payloads(blob["pages"])
            if page_raws:
                blob["text"] = self._extract_text_from_payloads(page_raws)

        return blob

    # Back-compat shim: pipeline expects this name
    def get_document_text(self, course_id: str, document_id: str) -> str:
        """
        Return cleaned text when available (prefer embedded 'text' produced by get_document_blob).
        """
        blob = self.get_document_blob(course_id, document_id)
        if isinstance(blob, dict):
            for k in ("text", "content", "plainText"):
                v = blob.get(k)
                if isinstance(v, str) and v.strip():
                    return v
        return ""  # upstream pipeline can fall back if needed

    # ----------- DEBUG/EXPORT ARTIFACTS ----------- #

    def export_debug_artifacts(self, course_id: str, document_id: str, out_dir: str) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)

        # (1) Library JSON
        url = f"/courses/{course_id}/library/{document_id}"
        try:
            r = self.client.get(url)
        except httpx.HTTPError:
            raise RuntimeError("Failed to contact Perusall library endpoint")

        if r.status_code == 403:
            raise RuntimeError("Forbidden or catalog-protected document; cannot export.")
        if r.status_code == 404:
            raise RuntimeError("Document not found; cannot export.")
        if r.status_code != 200:
            raise RuntimeError(f"Unexpected status {r.status_code} from library endpoint.")

        try:
            library_blob = r.json()
        except Exception as e:
            raise RuntimeError(f"Library response was not JSON: {e}")

        path1 = os.path.join(out_dir, "perusall_data.json")
        with open(path1, "w", encoding="utf-8") as f:
            json.dump(library_blob, f, ensure_ascii=False, indent=2)

        # (2) Concatenated raw payloads from all pages' textContentUrl
        raw_payloads: List[str] = []
        if isinstance(library_blob, dict) and isinstance(library_blob.get("pages"), list):
            raw_payloads = self._fetch_page_payloads(library_blob["pages"])

        # Join with an explicit page delimiter for readability/debugging
        # (This matches the spirit of the linked file: a dump of JSON from textContentUrls.)
        concatenated_raw = ""
        for idx, raw in enumerate(raw_payloads, start=1):
            if idx > 1:
                concatenated_raw += "\n\n" + ("=" * 80) + f"\n-- PAGE {idx} --\n" + ("=" * 80) + "\n\n"
            concatenated_raw += raw

        path2 = os.path.join(out_dir, "perusall_data_extracted.txt")
        with open(path2, "w", encoding="utf-8") as f:
            f.write(concatenated_raw)

        # (3) Cleaned pure text
        cleaned_text = self._extract_text_from_payloads(raw_payloads) if raw_payloads else ""
        path3 = os.path.join(out_dir, "perusall_data_extracted_cleaned.txt")
        with open(path3, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        return {"library_json": path1, "raw_extracted": path2, "cleaned_text": path3}

    # ----------- CREATE ----------- #

    def create_annotation(
        self,
        course_id: str,
        assignment_id: str,
        document_id: str,
        user_id: str,
        fragment: str,
        text: str,
        range_start: int,
        range_end: int,
        *,
        range_page: int = 1,
        range_type: str = "text",
        position_start_x: float = 0.0,
        position_start_y: float = 1.0,
        position_end_x: float = 0.48,
        position_end_y: float = 1.67,
    ) -> Dict[str, Any]:
        """
        POST an annotation to Perusall (unchanged).
        """
        payload = {
            "documentId": document_id,
            "userId": user_id,
            "positionStartX": position_start_x,
            "positionStartY": position_start_y,
            "positionEndX": position_end_x,
            "positionEndY": position_end_y,
            "rangeType": range_type,
            "rangePage": range_page,
            "rangeStart": range_start,
            "rangeEnd": range_end,
            "fragment": fragment,
            "text": text,
        }
        url = f"/courses/{course_id}/assignments/{assignment_id}/annotations"
        resp = self.client.post(url, data=payload)
        resp.raise_for_status()
        return resp.json()
