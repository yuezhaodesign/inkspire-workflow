# inkspire/generator.py
# Multi-agent flow (A/B/C) with LangChain + Gemini 2.5 Flash
# Public API: generate_annotations(material_text, kb_texts, provider="gemini")
# Guarantees:
# - Returns list of dicts with at least {"target_span", "text"}
# - Robust fallback when LLM/deps are unavailable
# - Attempts to map LLM spans back to exact substrings in source text
# - Avoids acknowledgments/author lists as scaffold targets

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import os
import json
import re
import unicodedata

# ---- Import robust locator (single source of truth for anchoring) ----
try:
    from .locator import find_first_occurrence  # package import
except Exception:
    from locator import find_first_occurrence    # fallback for script runs

# ---- LangChain imports (graceful availability flag) ----
_LC_AVAILABLE = True
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    _LC_AVAILABLE = False


# ---------------------------- Utilities ----------------------------

def _dbg_enabled() -> bool:
    return os.getenv("INKSPIRE_DEBUG", "0").lower() in {"1", "true"}

def _dbg(msg: str) -> None:
    if _dbg_enabled():
        try:
            print(f"[DEBUG] {msg}")
        except Exception:
            pass

# simple sentence splitter used throughout (dependency-free)
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+|\n+')

def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]

def _looks_like_ack(span: str) -> bool:
    """Heuristic: ack/author list/email-ish spans we want to avoid."""
    if not span:
        return False
    if span.count(",") >= 6:
        return True
    if "@" in span:
        return True
    # many Proper Names pattern
    if re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+", span) and span.count(",") >= 3:
        return True
    # affiliation clues
    if re.search(r"\bUniversity|Google|Zurich|USA|Canada|Affiliation|Acknowledg(e)?ments?\b", span, re.I):
        return True
    return False

def _trim_span_to_reasonable_window(src: str, exact: str,
                                    min_len: int = 120,
                                    max_len: int = 300) -> str:
    """
    If an exact slice is too short/long, expand/trim to sentence-ish window.
    """
    if not exact:
        return exact
    if min_len <= len(exact) <= max_len:
        return exact

    idx = (src or "").find(exact)
    if idx == -1:
        return exact

    sentences = _split_sentences(src)
    # rebuild offsets to find sentence boundaries covering [idx, idx+len(exact))
    offsets, pos = [], 0
    for s in sentences:
        start = src.find(s, pos)
        if start == -1:
            start = pos
        end = start + len(s)
        offsets.append((start, end))
        pos = end

    start_idx = idx
    end_idx = idx + len(exact)

    lb, rb = start_idx, end_idx
    for (s0, s1) in offsets:
        if s1 >= start_idx:
            lb = s0
            break
    for (s0, s1) in offsets:
        if s0 > end_idx:
            break
        rb = s1

    slice_ = src[lb:rb].strip()
    if len(slice_) > max_len:
        slice_ = slice_[:max_len].rsplit(" ", 1)[0] + "..."
    elif len(slice_) < min_len:
        pad = (min_len - len(slice_)) // 2
        lb2 = max(0, lb - pad)
        rb2 = min(len(src), rb + pad)
        slice_ = src[lb2:rb2].strip()
    return slice_


# ---------------------------- State ----------------------------

@dataclass
class PipelineState:
    material_text: str
    kb_texts: List[str] = field(default_factory=list)
    provider: str = "gemini"
    # Outputs
    material_summary: str = ""                          # Agent A
    candidate_spans: List[str] = field(default_factory=list)  # Agent B
    scaffolds: List[Dict[str, Any]] = field(default_factory=list)  # Agent C


# --------------------- Heuristic fallback agents ---------------------

def _heuristic_agent_a(state: PipelineState) -> None:
    text = (state.material_text or "").strip()
    sentences = _split_sentences(text)
    state.material_summary = (
        f"Material length: {len(text)} chars; ~{len(sentences)} sentences. "
        "No LLM: heuristic summary only."
    )

def _score_sentence(s: str) -> float:
    # Favor 80–300 chars and some punctuation density
    L = len(s)
    width = 1.0 - abs((L - 190) / 160)   # peak near 190
    punct = s.count(",") + s.count(";")
    return width + 0.2 * punct

def _heuristic_agent_b(state: PipelineState) -> None:
    src = state.material_text or ""
    sentences = _split_sentences(src)
    ranked = sorted((s for s in sentences if not _looks_like_ack(s)),
                    key=_score_sentence, reverse=True)
    picks = [s for s in ranked if len(s) >= 70][:3]
    if not picks and ranked:
        picks = [ranked[0]]

    mapped: List[str] = []
    for s in picks:
        rs, end_idx = find_first_occurrence(src, s)
        if rs != -1:
            exact = src[rs:end_idx]
            mapped.append(_trim_span_to_reasonable_window(src, exact))
        else:
            mapped.append(s)
    state.candidate_spans = mapped[:3]

def _heuristic_agent_c(state: PipelineState) -> None:
    out: List[Dict[str, Any]] = []
    src = state.material_text or ""
    for i, span in enumerate(state.candidate_spans, start=1):
        rs, end_idx = find_first_occurrence(src, span)
        if rs != -1:
            exact = src[rs:end_idx]
            windowed = _trim_span_to_reasonable_window(src, exact)
            anchor = {"start": rs, "end": end_idx, "fragment": exact}
        else:
            windowed = span
            anchor = {"start": -1, "end": -1, "fragment": ""}

        body = (
            f"### Scaffold #{i}\n"
            f"**Target Passage:** {windowed}\n\n"
            f"**Prompts:**\n"
            f"1) Paraphrase the passage in your own words (2–3 sentences).\n"
            f"2) Provide a micro-example (3–5 lines) illustrating the idea.\n"
            f"3) Identify one limitation, assumption, or failure case.\n"
        )
        out.append({
            "title": f"Scaffold #{i}",
            "target_span": windowed,
            "text": body,
            "anchor": anchor
        })
    state.scaffolds = out


# --------------------- Gemini-backed agents (LangChain) ---------------------

def _build_gemini(model_name: str = "gemini-2.5-flash", temperature: float = 0.2):
    if not _LC_AVAILABLE:
        raise RuntimeError("LangChain or langchain-google-genai not available.")
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set for Gemini.")
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

def _agent_a_llm(llm: "ChatGoogleGenerativeAI"):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are Agent A (Material Characterization). Read the provided text and produce:\n"
         "1) A 1-2 sentence high-level summary.\n"
         "2) 3-6 salient key terms (lowercase, comma separated).\n"
         "Return JSON with fields: summary: str, keywords: list[str]."),
        ("human", "{material_text}")
    ])
    chain = prompt | llm | StrOutputParser()

    def _run(state: PipelineState) -> PipelineState:
        raw = chain.invoke({"material_text": state.material_text[:12000]})
        try:
            data = json.loads(raw)
            summary = str(data.get("summary", "")).strip()
        except Exception:
            summary = raw.strip().strip("`")
        state.material_summary = summary[:2000]
        return state

    return RunnableLambda(_run)

def _agent_b_llm(llm: "ChatGoogleGenerativeAI"):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are Agent B (Scaffold-Worthy Area Identification). From the input text, pick 1-3 passages that are challenging or central.\n"
         "CRITICAL:\n"
         "- Return exact substrings copied verbatim from the input (no rewriting).\n"
         "- Prefer longer, information-dense spans (80-300 chars).\n"
         "- Avoid acknowledgments, author lists, affiliations, emails, and references sections.\n"
         "Return a JSON array of strings, e.g. [\"span1\", \"span2\"]."),
        ("human", "{material_text}")
    ])
    chain = prompt | llm | StrOutputParser()

    def _run(state: PipelineState) -> PipelineState:
        src = state.material_text
        raw = chain.invoke({"material_text": src[:40000]})
        spans: List[str] = []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                spans = [s for s in data if isinstance(s, str)]
        except Exception:
            # salvage quoted lines if any
            spans = re.findall(r'"([^"]{40,500})"', raw)[:3]

        # post-filter: drop ack-looking spans
        spans = [s for s in spans if not _looks_like_ack(s)]

        # map to exact substrings in src (using robust locator)
        mapped: List[str] = []
        for s in spans:
            rs, end_idx = find_first_occurrence(src, s)
            if rs != -1:
                exact = src[rs:end_idx]
                mapped.append(_trim_span_to_reasonable_window(src, exact))
        if not mapped:
            _heuristic_agent_b(state)
        else:
            state.candidate_spans = mapped[:3]
        return state

    return RunnableLambda(_run)

def _agent_c_llm(llm: "ChatGoogleGenerativeAI"):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are Agent C (Annotation-Based Scaffold Generation).\n"
         "For each TARGET SPAN, create a short scaffold (markdown) with exactly these prompts:\n"
         "1) Paraphrase the passage in your own words (2–3 sentences).\n"
         "2) Provide a micro-example (3–5 lines) illustrating the idea.\n"
         "3) Identify one limitation, assumption, or failure case.\n"
         # Escape braces so LangChain doesn't treat them as variables:
         "Return JSON array of objects: [{{\"target_span\": string, \"markdown\": string}}].\n"
         "IMPORTANT: The braces above are a literal JSON example; do not treat them as variables."
        ),
        ("human",
         "TARGET SPANS (JSON):\n{spans_json}\n\n"
         "GLOBAL CONTEXT (truncated):\n{material_head}")
    ])
    chain = prompt | llm | StrOutputParser()

    def _run(state: PipelineState) -> PipelineState:
        spans = state.candidate_spans[:3]
        src = state.material_text
        raw = chain.invoke({
            "spans_json": json.dumps(spans, ensure_ascii=False),
            "material_head": src[:1200]
        })
        out: List[Dict[str, Any]] = []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    span = str(item.get("target_span", "")).strip()
                    md = str(item.get("markdown", "")).strip()
                    if not span or not md:
                        continue

                    rs, end_idx = find_first_occurrence(src, span)
                    if rs == -1:
                        continue
                    exact = src[rs:end_idx]
                    windowed = _trim_span_to_reasonable_window(src, exact)

                    out.append({
                        "title": "Scaffold",
                        "target_span": windowed,
                        "text": md,
                        "anchor": {"start": rs, "end": end_idx, "fragment": exact},
                    })
        except Exception:
            _heuristic_agent_c(state)
            return state

        if not out:
            _heuristic_agent_c(state)
        else:
            state.scaffolds = out
        return state

    return RunnableLambda(_run)


# --------------------------- Orchestration ---------------------------

def _run_llm_pipeline(state: PipelineState, model_name: str = "gemini-2.5-flash") -> PipelineState:
    llm = _build_gemini(model_name=model_name, temperature=0.2)
    a = _agent_a_llm(llm)
    b = _agent_b_llm(llm)
    c = _agent_c_llm(llm)
    s = a.invoke(state)
    s = b.invoke(s)
    s = c.invoke(s)
    return s

def _run_heuristic_pipeline(state: PipelineState) -> PipelineState:
    _heuristic_agent_a(state)
    _heuristic_agent_b(state)
    _heuristic_agent_c(state)
    return state


# ---------------------------- Public API ----------------------------

def generate_annotations(material_text: str, kb_texts: List[str], provider: str = "gemini") -> List[Dict[str, Any]]:
    """
    Entry used by pipeline.run().
    Returns a list of annotations, each with {"target_span", "text"}.
    Uses Gemini 2.5 Flash via LangChain; falls back heuristically if unavailable.
    """
    state = PipelineState(
        material_text=material_text or "",
        kb_texts=kb_texts or [],
        provider=(provider or "gemini").lower(),
    )

    # Handle ultra-short inputs gracefully
    if len(state.material_text.strip()) < 40:
        return []

    used_llm = True
    try:
        state = _run_llm_pipeline(state, model_name="gemini-2.5-flash")
    except Exception as e:
        used_llm = False
        _dbg(f"Falling back to heuristic pipeline due to error: {e}")
        state = _run_heuristic_pipeline(state)

    anns = state.scaffolds[:8]

    # Add provenance hint if fallback used (non-intrusive)
    if not used_llm:
        for a in anns:
            a.setdefault("meta", {})["note"] = "Generated via heuristic fallback — Gemini unavailable"

    return anns


# ----------------------- Optional local demo -----------------------

if __name__ == "__main__":
    demo = (
        "This section defines a simple algorithm for stable matching. "
        "The algorithm proceeds by repeatedly pairing unmatched participants until a fixed point is reached. "
        "We provide a proof sketch that the process terminates and yields a stable outcome under standard assumptions. "
        "As an example, consider three proposers and three receivers with strict preferences; the algorithm converges in at most nine steps."
    )
    res = generate_annotations(demo, kb_texts=[], provider="gemini")
    print(f"Generated {len(res)} annotations (showing target spans & first lines):")
    for i, r in enumerate(res, 1):
        print(f"\n[{i}] target_span: {r['target_span'][:80]}{'...' if len(r['target_span'])>80 else ''}")
        print(r.get('text', '').splitlines()[0] if r.get('text') else "")
        if 'anchor' in r:
            print(f"  -> anchor: {r['anchor'].get('start')}–{r['anchor'].get('end')}")
