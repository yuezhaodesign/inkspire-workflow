from __future__ import annotations
import os, re, math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

# ---------- tiny utils ----------
_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(s or "")]

def _chunk_text(text: str, max_chars: int = 1500, overlap: int = 300) -> List[str]:
    text = text or ""
    if len(text) <= max_chars: return [text]
    chunks: List[str] = []; i = 0
    while i < len(text):
        j = min(len(text), i + max_chars); segment = text[i:j]; k = segment.rfind("\\n\\n")
        if k > max_chars * 0.5: segment = segment[:k]; j = i + k
        chunks.append(segment)
        if j == len(text): break
        i = max(0, j - overlap)
    return chunks

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\\s+|\\n{1,2}', text or "")
    out = []
    for p in parts:
        p = p.strip()
        if len(p) >= 30 and any(ch.isalpha() for ch in p):
            out.append(p)
    return out

def _extract_code_blocks(text: str) -> List[str]:
    blocks = []
    for m in re.finditer(r"```[a-zA-Z]*\\n(.*?)```", text or "", flags=re.S):
        b = m.group(1).strip()
        if len(b) >= 8: blocks.append(b)
    for m in re.finditer(r"(?:^|\\n)( {4,}.*(?:\\n {4,}.*)+)", text or "", flags=re.S):
        b = m.group(1).strip()
        if len(b) >= 8: blocks.append(b)
    return blocks[:12]

def _build_tfidf_index(docs: List[str]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    N = len(docs); dfs: Dict[str, int] = {}; tfs: List[Dict[str, int]] = []
    for d in docs:
        toks = _tokenize(d); tf: Dict[str, int] = {}; seen = set()
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
            if t not in seen: dfs[t] = dfs.get(t, 0) + 1; seen.add(t)
        tfs.append(tf)
    idf = {t: math.log((N + 1) / (df + 1)) + 1.0 for t, df in dfs.items()}
    tfidf_docs = [{t: tf[t] * idf.get(t, 0.0) for t in tf} for tf in tfs]
    return tfidf_docs, idf

def _tfidf_vectorize(query: str, idf: Dict[str, float]) -> Dict[str, float]:
    tf: Dict[str, int] = {}
    for t in _tokenize(query): tf[t] = tf.get(t, 0) + 1
    return {t: tf[t] * idf.get(t, 0.0) for t in tf}

def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b: return 0.0
    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in a.keys() & b.keys())
    na = math.sqrt(sum(v * v for v in a.values())); nb = math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

def _retrieve(query: str, corpus: List[str], tfidf_docs: List[Dict[str, float]], idf: Dict[str, float], k: int = 6) -> List[int]:
    qv = _tfidf_vectorize(query, idf)
    scores = [(i, _cosine(qv, tfidf_docs[i])) for i in range(len(corpus))]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, s in scores[:k] if s > 0]

def _strip_front_matter(s: str) -> str:
    if not s: return s
    lines = s.splitlines()
    head = "\\n".join(lines[:120]); body = "\\n".join(lines[120:])
    head = re.sub(r"(?:^[A-Z][A-Z \\-\\.]+(?:\\n|$)){2,}", "", head, flags=re.M)
    return (head + "\\n" + body).strip()

def _strip_references(s: str) -> str:
    if not s: return s
    m = re.search(r"\\n\\s*(References|Bibliography)\\s*\\n", s, flags=re.I)
    return s[:m.start()].strip() if m else s

# ---------- KB ingestion ----------
def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _ingest_kb(kb_texts: List[str]) -> List[str]:
    texts: List[str] = []

    # 1) CLI-provided strings can be file/dir paths or in-memory text
    for entry in kb_texts or []:
        if not entry: continue
        p = Path(entry)
        if p.exists():
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.suffix.lower() in {".txt", ".md", ".markdown"} and f.is_file():
                        t = _read_text_file(f)
                        if t.strip(): texts.append(t)
            elif p.is_file():
                if p.suffix.lower() in {".txt", ".md", ".markdown"}:
                    t = _read_text_file(p)
                    if t.strip(): texts.append(t)
        else:
            # treat as raw text if it's long enough
            if len(entry) > 80:
                texts.append(entry)

    # 2) Env var override
    kb_dir_env = os.getenv("INKSPIRE_KB_DIR", "").strip()
    if kb_dir_env:
        p = Path(kb_dir_env)
        if p.exists() and p.is_dir():
            for f in p.rglob("*"):
                if f.suffix.lower() in {".txt", ".md", ".markdown"} and f.is_file():
                    t = _read_text_file(f)
                    if t.strip(): texts.append(t)

    # 3) Default ./kb_folder
    default_dir = Path("kb_folder")
    if default_dir.exists() and default_dir.is_dir():
        for f in default_dir.rglob("*"):
            if f.suffix.lower() in {".txt", ".md", ".markdown"} and f.is_file():
                t = _read_text_file(f)
                if t.strip(): texts.append(t)

    return texts

# ---------- LangGraph state ----------
@dataclass
class PipelineState:
    mode: Literal["annotations","scaffolds"]
    full_text: str
    kb_texts: List[str]
    provider: str = "openai"
    chunks: List[str] = field(default_factory=list)
    kb_chunks: List[str] = field(default_factory=list)
    corpus: List[str] = field(default_factory=list)
    tfidf_docs: List[Dict[str, float]] = field(default_factory=list)
    idf: Dict[str, float] = field(default_factory=dict)
    targets: List[Dict[str, Any]] = field(default_factory=list)
    final_items: List[Dict[str, Any]] = field(default_factory=list)

try:
    from langgraph.graph import StateGraph, END
except Exception:
    StateGraph = None
    END = "__END__"

# ---------- Agent A ----------
def node_extract(state: PipelineState) -> PipelineState:
    # Ingest KB robustly from kb_texts/env/default
    kb_all = _ingest_kb(state.kb_texts)

    clean = _strip_references(_strip_front_matter(state.full_text))
    state.chunks = _chunk_text(clean, 1500, 300)
    sentences = _split_sentences(clean)[:600]
    code_blocks = _extract_code_blocks(clean)

    # Build KB chunks
    kb_chunks: List[str] = []
    for kb in kb_all:
        kb_chunks.extend(_chunk_text(kb, 1000, 100))
    state.kb_chunks = kb_chunks[:300]

    # Index
    state.corpus = state.chunks + state.kb_chunks
    if not state.corpus:
        state.corpus = [clean] if clean else [""]
    state.tfidf_docs, state.idf = _build_tfidf_index(state.corpus)

    # Build KB centroid if KB exists
    kb_vecs = state.tfidf_docs[len(state.chunks):] if state.kb_chunks else []
    centroid: Dict[str, float] = {}
    for v in kb_vecs:
        for t, w in v.items():
            centroid[t] = centroid.get(t, 0.0) + w

    ranked: List[Tuple[float, str, str]] = []  # (score, type, anchor)

    if centroid:
        # RAG-driven ranking
        for s in sentences:
            sv = _tfidf_vectorize(s, state.idf); sc = _cosine(sv, centroid)
            if sc > 0 and 20 <= len(s) <= 240: ranked.append((sc, "Conceptual", s))
        for b in code_blocks:
            for ln in [ln for ln in b.splitlines() if ln.strip()][:3]:
                sv = _tfidf_vectorize(ln, state.idf); sc = _cosine(sv, centroid)
                if sc > 0 and 8 <= len(ln) <= 200: ranked.append((sc * 1.1, "Code", ln))
    else:
        # Fallback ranking (no KB): choose high-information sentences / code
        def info_score(s: str) -> float:
            toks = _tokenize(s)
            alpha = sum(ch.isalpha() for ch in s)
            num = sum(ch.isdigit() for ch in s)
            uniq = len(set(toks))
            return 0.6*uniq + 0.3*(alpha/ max(1,len(s))) + 0.1*min(num,10)
        for s in sentences:
            if 20 <= len(s) <= 240:
                ranked.append((info_score(s), "Conceptual", s))
        for b in code_blocks:
            for ln in [ln for ln in b.splitlines() if ln.strip()][:3]:
                if 8 <= len(ln) <= 200:
                    ranked.append((info_score(ln)*1.1, "Code", ln))

    ranked.sort(key=lambda x: x[0], reverse=True)
    picked: List[Tuple[str, str]] = []
    seen = set()
    for _, ctype, anchor in ranked:
        if anchor in seen: continue
        seen.add(anchor)
        picked.append((ctype, anchor))
        if len(picked) >= 8: break

    targets: List[Dict[str, Any]] = []
    for ctype, a in picked:
        targets.append({
            "anchor_cue": a,
            "intent": "question" if a.endswith("?") else "comment",
            "content_type": ctype,
            "priority": "High",
        })
    state.targets = targets
    return state

# ---------- Agent B ----------
def _retrieve_indices(anchor: str, state: PipelineState) -> List[int]:
    try:
        return _retrieve(anchor, state.corpus, state.tfidf_docs, state.idf, k=6)
    except Exception:
        return []

def _nearest_kb(anchor: str, state: PipelineState) -> str:
    idxs = _retrieve_indices(anchor, state)
    kb_offset = len(state.chunks)
    kb_indices = [i for i in idxs if i >= kb_offset]
    if kb_indices:
        return state.corpus[kb_indices[0]]
    return state.corpus[idxs[0]] if idxs else ""

def node_generate(state: PipelineState) -> PipelineState:
    items: List[Dict[str, Any]] = []
    for t in state.targets:
        anchor = t.get("anchor_cue","").strip()
        if not anchor: continue
        kb_hit = _nearest_kb(anchor, state)
        kb_first = kb_hit.splitlines()[0].strip() if kb_hit else ""

        if state.mode == "annotations":
            typ = "question" if t.get("intent") == "question" else "comment"
            body = (
                (f"What does this mean in your own words? Connect it to: {kb_first[:120]}. "
                 "Show a tiny example or quick check.") if typ == "question"
                else (f"Key connection: {kb_first[:140]}. "
                      "Consider inputs, state changes, and outputs implied here.")
            )
            items.append({"anchor_text": anchor, "type": typ, "text": body})
        else:
            content_type = t.get("content_type") or "Conceptual"
            cog = "variable tracing" if content_type == "Code" else "conceptual understanding"
            md = []
            md.append(f"### Scaffold: Working with “{anchor[:60]}”")
            md.append(f"**Target Content:** exact anchor in text\n**Content Type:** {content_type}\n**Cognitive Focus:** {cog}")
            md.append("**Scaffold Prompt:**\n- Paraphrase the idea.\n- Create a micro-example (3–5 lines).\n- Explain why it works or when it fails.")
            md.append("**Strategy Foundation:** PRIMM-Investigate\n**Expected Engagement Time:** 5-10 minutes\n**Collaboration Mode:** Individual")
            md.append(f"**Learning Objective:** Connect the passage to {kb_first[:120]}")
            md.append("**Instructor Note:** Look for alignment between code and explanation.")
            markdown = "\\n\\n".join(md)
            items.append({
                "anchor_text": anchor,
                "content_type": content_type,
                "cognitive_focus": cog,
                "differentiation": "Standard",
                "expected_time": "5-10 minutes",
                "collaboration_mode": "Individual",
                "markdown": markdown,
            })

    # dedupe by anchor
    out: List[Dict[str, Any]] = []
    seen = set()
    for it in items:
        a = it.get("anchor_text","")
        if a in seen: continue
        seen.add(a)
        out.append(it)
    state.final_items = out[:8]
    return state

# ---------- Graph & API ----------
def _compile_graph():
    try:
        from langgraph.graph import StateGraph, END
    except Exception:
        return None
    graph = StateGraph(PipelineState)
    graph.add_node("AgentA_Extract", node_extract)
    graph.add_node("AgentB_Generate", node_generate)
    graph.set_entry_point("AgentA_Extract")
    graph.add_edge("AgentA_Extract", "AgentB_Generate")
    graph.add_edge("AgentB_Generate", END)
    return graph.compile()

def _run_pipeline(mode: Literal["annotations","scaffolds"], full_text: str, kb_texts: List[str], provider: str) -> List[Dict]:
    st = PipelineState(mode=mode, full_text=full_text or "", kb_texts=kb_texts or [], provider=provider or "openai")
    app = _compile_graph()
    if app is None:
        st = node_extract(st); st = node_generate(st); return st.final_items
    res = app.invoke(st)  # type: ignore
    if isinstance(res, dict): return res.get("final_items", []) or []
    return getattr(res, "final_items", []) or []

def generate_annotations(full_text: str, kb_texts: List[str], provider: str = "openai") -> List[Dict]:
    items = _run_pipeline("annotations", full_text, kb_texts, provider)
    return [ {"anchor_text": it.get("anchor_text",""), "type": it.get("type","question"), "text": it.get("text","")} for it in items ][:8]

def generate_scaffolds(full_text: str, kb_texts: List[str], provider: str = "openai") -> List[Dict]:
    items = _run_pipeline("scaffolds", full_text, kb_texts, provider)
    out = [{
        "anchor_text": it.get("anchor_text",""),
        "content_type": it.get("content_type","Conceptual"),
        "cognitive_focus": it.get("cognitive_focus","conceptual understanding"),
        "differentiation": it.get("differentiation","Standard"),
        "expected_time": it.get("expected_time","5-10 minutes"),
        "collaboration_mode": it.get("collaboration_mode","Individual"),
        "markdown": it.get("markdown",""),
    } for it in items]
    if len(out) < 5: out = (out * 5)[:5]
    return out[:8]
