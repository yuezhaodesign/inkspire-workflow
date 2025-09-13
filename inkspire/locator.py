# locator.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import re
import unicodedata


def _normalize_with_map(s: str):
    """
    Build a normalized string plus a mapping from normalized index -> original index.
    Normalization: NFKC, lowercase, curly->straight quotes, collapse whitespace runs to ' ',
    drop control chars (e.g., from PDFs).
    """
    out = []
    idx_map = []
    i = 0
    N = len(s)
    while i < N:
        ch = s[i]
        ch_n = unicodedata.normalize("NFKC", ch)

        # drop control characters entirely
        if unicodedata.category(ch_n) == "Cc":
            i += 1
            continue

        # unify quotes
        if ch_n in ("“", "”"):
            ch_n = '"'
        elif ch_n in ("‘", "’"):
            ch_n = "'"

        if ch_n.isspace():
            # collapse whitespace/control runs
            j = i + 1
            while j < N:
                nxt = unicodedata.normalize("NFKC", s[j])
                if nxt.isspace() or unicodedata.category(nxt) == "Cc":
                    j += 1
                else:
                    break
            out.append(" ")
            idx_map.append(j - 1)  # map to last original index in the run
            i = j
            continue

        out.append(ch_n.lower())
        idx_map.append(i)
        i += 1

    return "".join(out), idx_map

def _tolerant_search(norm_hay: str, norm_needle: str) -> Optional[Tuple[int, int]]:
    """
    Allow small punctuation/whitespace gaps between tokens, useful when spacing differs.
    Returns (start, end) in normalized coordinates, or None.
    """
    toks = norm_needle.split()
    if len(toks) < 2:
        pos = norm_hay.find(norm_needle)
        return (pos, pos + len(norm_needle)) if pos != -1 else None

    # Allow spaces OR up to 3 punctuation chars between tokens
    pat = r"(?:\s+|[^\w\s]{0,3})".join(re.escape(t) for t in toks)
    m = re.search(pat, norm_hay, flags=re.DOTALL)
    if not m:
        return None
    return (m.start(), m.end())

def find_first_occurrence(haystack: str, needle: str) -> Tuple[int, int]:
    """
    Return (start, end) char offsets for the first occurrence of `needle` in `haystack`.
    Robust to curly/straight quotes, whitespace runs, case, and minor punctuation gaps.
    Returns (-1, -1) when not found or inputs are empty.
    """
    if not haystack or not needle:
        return (-1, -1)

    norm_hay, map_hay = _normalize_with_map(haystack)
    norm_needle, _ = _normalize_with_map(needle)

    # 1) direct normalized substring
    pos = norm_hay.find(norm_needle)
    if pos != -1:
        s0 = map_hay[pos]
        s1 = map_hay[min(pos + len(norm_needle) - 1, len(map_hay) - 1)] + 1
        return (s0, s1)

    # 2) tolerant token-gap search
    span = _tolerant_search(norm_hay, norm_needle)
    if span:
        n0, n1 = span
        s0 = map_hay[n0]
        s1 = map_hay[min(n1 - 1, len(map_hay) - 1)] + 1
        return (s0, s1)

    return (-1, -1)

def attach_text_ranges(cleaned_text: str, anns: List[Dict]) -> List[Dict]:
    """
    For each annotation, compute: rangeStart, rangeEnd, fragment.
    Looks for 'anchor_text' first, then falls back to 'target_span'.
    Leaves rangeStart/rangeEnd as -1 and fragment "" if not found.
    """
    out: List[Dict] = []
    for a in anns:
        anchor = (a.get("anchor_text") or a.get("target_span") or "").strip()
        rs, re = find_first_occurrence(cleaned_text, anchor) if anchor else (-1, -1)
        frag = cleaned_text[rs:re] if (0 <= rs < re <= len(cleaned_text)) else ""
        out.append({
            "positionStartX": a.get("positionStartX", 0.0),
            "positionStartY": a.get("positionStartY", 1.0),
            "positionEndX":   a.get("positionEndX", 0.48),
            "positionEndY":   a.get("positionEndY", 1.67),
            "rangeType": "text",
            "rangePage": a.get("rangePage", 1),
            "rangeStart": rs,
            "rangeEnd": re,
            "fragment": frag,
            "text": a.get("text", ""),
            "_anchor_text": anchor,
            "_type": a.get("type", "comment"),
        })
    return out
