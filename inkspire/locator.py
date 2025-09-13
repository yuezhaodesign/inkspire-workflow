# inkspire/locator.py
from __future__ import annotations
import re
import unicodedata
from typing import Tuple, List, Dict, Optional
from difflib import SequenceMatcher

__all__ = [
    "find_first_occurrence",
    "find_in_pages",
    "attach_text_ranges",
]

# ---------------- Normalization ----------------

_ZW = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]")  # zero-width & bidi
_SOFT_HYPHEN = "\u00ad"
_LIGATURES = {"ﬁ":"fi","ﬂ":"fl","ﬀ":"ff","ﬃ":"ffi","ﬄ":"ffl"}

def _normalize_quotes_dashes(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    return (s
        .replace("“", '"').replace("”", '"').replace("„", '"').replace("‟", '"')
        .replace("‘", "'").replace("’", "'").replace("‚", "'")
        .replace("–", "-").replace("—", "-").replace("−", "-")
        .replace(_SOFT_HYPHEN, "")
    )

def _strip_zero_width_and_ligatures(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = _ZW.sub("", s)
    for k, v in _LIGATURES.items():
        s = s.replace(k, v)
    return s

def _canonical(s: str) -> str:
    s = _strip_zero_width_and_ligatures(s)
    s = unicodedata.normalize("NFKC", s)
    s = _normalize_quotes_dashes(s)
    return s

# ---------------- Flexible regex ----------------

def _tokenize(needle: str) -> List[str]:
    n = _canonical(needle or "").strip()
    return [t for t in re.split(r"\s+", n) if t]

def _escape_token_flex(tok: str) -> str:
    e = re.escape(tok)
    e = e.replace(r"\"", r"[" + '"' + "“”" + r"]")
    e = e.replace(r"\'", r"[\'‘’]")
    e = e.replace(r"\-", r"[-–—−" + _SOFT_HYPHEN + r"]")
    return e

def _charflex(token_pattern: str) -> str:
    """
    Turn a token pattern into a character-flex version that allows optional whitespace
    between characters. Example: 'addition' -> a\s*d\s*d\s*i\s*t\s*i\s*o\s*n (with the
    token's own quote/dash flex preserved).
    """
    # Remove outer escapes for per-char split while preserving classes.
    # Simplest robust approach: expand only alnum segments per char.
    parts = []
    i = 0
    while i < len(token_pattern):
        ch = token_pattern[i]
        if ch == "\\" and i + 1 < len(token_pattern):
            # escaped literal
            parts.append(token_pattern[i:i+2])
            i += 2
        elif ch == "[":
            j = token_pattern.find("]", i+1)
            if j != -1:
                parts.append(token_pattern[i:j+1])
                i = j + 1
            else:
                parts.append(re.escape(ch)); i += 1
        else:
            parts.append(re.escape(ch))
            i += 1
    # join with optional whitespace between chars/classes
    return r"\s*".join(parts)

def _build_relaxed_regex(needle: str, allow_charflex: bool = False) -> re.Pattern:
    """
    Case-insensitive, DOTALL regex:
      - tokens in order
      - between tokens: whitespace/newlines, optional dash (handles hyphen wraps)
      - inside tokens: flexible quotes/dashes; optional per-char whitespace if allow_charflex
    """
    toks = _tokenize(needle)
    if not toks:
        return re.compile(r"(?!)")
    escaped = [_escape_token_flex(t) for t in toks]
    if allow_charflex:
        escaped = [_charflex(p) for p in escaped]
    between = r"(?:\s*[-–—−" + _SOFT_HYPHEN + r"]?\s+)"
    pattern = r"(?s)" + between.join(escaped)
    return re.compile(pattern, flags=re.IGNORECASE)

# ---------------- Document / Page locators ----------------

def _try_patterns(text: str, needle: str) -> Tuple[int, int]:
    # 1) Exact
    s = text.find(needle)
    if s != -1: return (s, s + len(needle))
    s = text.lower().find(needle.lower())
    if s != -1: return (s, s + len(needle))
    # 2) Relaxed tokens
    for allow_charflex in (False, True):  # try normal first, then char-flex (for "s p a c e d")
        pat = _build_relaxed_regex(needle, allow_charflex=allow_charflex)
        m = pat.search(text)
        if m: return (m.start(), m.end())
    return (-1, -1)

def find_first_occurrence(haystack: str, needle: str) -> Tuple[int, int]:
    if not haystack or not needle:
        return (-1, -1)
    return _try_patterns(haystack, needle)

def _find_on_page(page_text: str, needle: str) -> Tuple[int, int]:
    if not page_text or not needle:
        return (-1, -1)
    rs, re_ = _try_patterns(page_text, needle)
    if rs != -1: return (rs, re_)
    # conservative fuzzy (only if long)
    if len(needle) >= 50:
        p_norm = _canonical(page_text)
        n_norm = _canonical(needle)
        step = max(25, len(needle)//6)
        best = (0.0, -1, -1)
        for i in range(0, max(1, len(p_norm)-len(n_norm)+1), step):
            seg = p_norm[i:i+len(n_norm)]
            r = SequenceMatcher(None, seg, n_norm).ratio()
            if r > best[0]:
                best = (r, i, i+len(n_norm))
        if best[0] >= 0.90:
            # try re-anchoring via relaxed regex on original page_text
            rs, re_ = _try_patterns(page_text, needle)
            if rs != -1:
                return (rs, re_)
    return (-1, -1)

def find_in_pages(pages: List[str], needle: str) -> Tuple[int, int, int]:
    if not pages or not needle:
        return (-1, -1, -1)
    # per-page
    for p_idx, txt in enumerate(pages):
        rs, re_ = _find_on_page(txt or "", needle)
        if rs != -1:
            return (p_idx + 1, rs, re_)
    # cross-page bridge (clip to the page where it starts)
    for p_idx in range(len(pages) - 1):
        left = pages[p_idx] or ""
        right = pages[p_idx+1] or ""
        bridge = f"{left}\n{right}"
        for allow_charflex in (False, True):
            pat = _build_relaxed_regex(needle, allow_charflex=allow_charflex)
            m = pat.search(bridge)
            if not m:
                continue
            s, e = m.start(), m.end()
            if s < len(left):
                return (p_idx + 1, s, min(e, len(left)))
            else:
                s2 = s - (len(left) + 1)
                e2 = max(s2 + 1, e - (len(left) + 1))
                if 0 <= s2 < len(right):
                    return (p_idx + 2, s2, min(e2, len(right)))
    return (-1, -1, -1)

# ---------------- Perusall range attachment ----------------

def attach_text_ranges(
    cleaned_text: str,
    anns: List[Dict],
    pages: Optional[List[str]] = None
) -> List[Dict]:
    """
    Build Perusall-ready ranges. If `pages` provided, offsets are within that page.
    """
    out: List[Dict] = []
    src = cleaned_text or ""
    page_list = pages if isinstance(pages, list) and pages else None

    def _safe_slice(s: str, i: int, j: int) -> str:
        if not isinstance(i, int) or not isinstance(j, int):
            return ""
        if i < 0 or j < 0 or j <= i or i >= len(s) or j > len(s):
            return ""
        return s[i:j]

    for a in anns or []:
        rs = re_ = -1
        frag = ""
        page_no = 1

        # prefer explicit anchor
        if isinstance(a.get("anchor"), dict):
            anchor = a["anchor"]
            try:
                rs = int(anchor.get("start", -1))
                re_ = int(anchor.get("end", -1))
                page_no = int(anchor.get("page", page_no))
            except Exception:
                rs, re_ = -1, -1
            if page_list and 1 <= page_no <= len(page_list):
                frag = _safe_slice(page_list[page_no - 1], rs, re_)
            else:
                frag = anchor.get("fragment") or _safe_slice(src, rs, re_)

        # else derive from anchor_text or target_span
        if rs == -1 or re_ == -1 or not frag:
            anchor_text = (a.get("anchor_text") or a.get("target_span") or "").strip()
            a["_anchor_text"] = anchor_text
            if anchor_text:
                if page_list:
                    page_no, rs, re_ = find_in_pages(page_list, anchor_text)
                    if page_no != -1:
                        frag = _safe_slice(page_list[page_no - 1], rs, re_)
                else:
                    rs, re_ = find_first_occurrence(src, anchor_text)
                    frag = _safe_slice(src, rs, re_)
                    page_no = 1 if rs != -1 else 1
            else:
                rs, re_, frag = -1, -1, ""
                page_no = 1

        out.append({
            "positionStartX": 0.0,
            "positionStartY": 1.0,
            "positionEndX": 0.48,
            "positionEndY": 1.67,
            "rangeType": "text",
            "rangePage": page_no if (isinstance(page_no, int) and page_no > 0) else 1,
            "rangeStart": rs if isinstance(rs, int) else -1,
            "rangeEnd": re_ if isinstance(re_, int) else -1,
            "fragment": frag or "",
            "text": a.get("text", ""),
            "_anchor_text": a.get("_anchor_text", ""),
            "_type": a.get("type", "comment"),
        })
    return out
