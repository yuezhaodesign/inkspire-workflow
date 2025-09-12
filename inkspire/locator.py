from __future__ import annotations
from typing import Dict, List, Tuple

def find_first_occurrence(haystack: str, needle: str) -> Tuple[int, int]:
    """
    Return (start, end) char offsets for the first occurrence of `needle` in `haystack`.
    Case-sensitive. Returns (-1, -1) if not found or if inputs are empty.
    """
    if not haystack or not needle:
        return (-1, -1)
    start = haystack.find(needle)
    if start == -1:
        return (-1, -1)
    return (start, start + len(needle))

def attach_text_ranges(cleaned_text: str, anns: List[Dict]) -> List[Dict]:
    """
    For each annotation (expects 'anchor_text' and 'text'), compute:
      rangeStart, rangeEnd, fragment
    If exact match is not found, leaves rangeStart/rangeEnd as -1 and fragment "".
    """
    out: List[Dict] = []
    for a in anns:
        anchor = (a.get("anchor_text") or "").strip()
        rs, re = find_first_occurrence(cleaned_text, anchor) if anchor else (-1, -1)
        frag = cleaned_text[rs:re] if rs >= 0 and re >= 0 else ""
        out.append({
            "positionStartX": 0.0,
            "positionStartY": 1.0,
            "positionEndX": 0.48,
            "positionEndY": 1.67,
            "rangeType": "text",
            "rangePage": 1,
            "rangeStart": rs,
            "rangeEnd": re,
            "fragment": frag,
            "text": a.get("text", ""),
            # Keep originals in case you need them later
            "_anchor_text": anchor,
            "_type": a.get("type", "comment"),
        })
    return out
