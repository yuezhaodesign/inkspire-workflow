from __future__ import annotations
import re
from typing import Any, Dict, List, Union

def _from_pdf_items_json(blob: Dict[str, Any]) -> str:
    """
    Handle Perusall/PDF text-layer style:
    {
      "items": [
        {"str": "AI-Assisted Assessment of Coding Practices", "x":..., "y":..., ...},
        {"str": "some next token", ...},
        ...
      ]
    }
    This simple pass concatenates item.str with spaces, then normalizes whitespace.
    If you want better paragraph reconstruction, you can group by y deltas.
    """
    items = blob.get("items") or []
    parts: List[str] = []
    for it in items:
        s = it.get("str")
        if isinstance(s, str):
            parts.append(s)
    text = " ".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text)
    return text.strip()

def clean_text(raw: Union[str, Dict[str, Any]]) -> str:
    """
    Normalize the document into plain text.
    - If raw is a dict with 'items', build text from items[].str
    - If raw is a string, normalize whitespace
    """
    if isinstance(raw, dict) and "items" in raw:
        text = _from_pdf_items_json(raw)
    elif isinstance(raw, str):
        text = raw
    else:
        return ""

    # Final normalization
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
