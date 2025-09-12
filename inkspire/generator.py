from __future__ import annotations
"""
Three-agent, prompt-driven LangGraph pipeline with KB RAG (LangChain + LangGraph).

Agents:
- AgentA_Analyze: Phase 1 Material Characterization (LLM with provided prompt)
- AgentB_Extract: Phase 2 Scaffold-Worthy Area Identification (LLM with provided prompt)
- AgentC_Generate: Phase 3 Annotation-Based Scaffold Generation (LLM with provided prompt)

RAG:
- Loads KB from paths in `kb_texts` (file/dir), env var INKSPIRE_KB_DIR, and default ./kb_folder
- Builds TF-IDF index over reading + KB chunks to provide nearest-context snippets to each agent

Public APIs:
- generate_annotations(full_text, kb_texts, provider="openai")  # Part 2 items only (back-compat)
- generate_scaffolds(full_text, kb_texts, provider="openai")   # alias
- generate_full_report(full_text, kb_texts, provider="openai", out_dir=None)
    -> returns {"overview_markdown","items"}; if out_dir is set writes overview.md & annotations.json
"""

import os, re, math, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# -----------------------------
# LangChain / LLM
# -----------------------------
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    try:
        from langchain_anthropic import ChatAnthropic
    except Exception:
        ChatAnthropic = None
except Exception:
    ChatPromptTemplate = None
    ChatOpenAI = None
    ChatAnthropic = None

# -----------------------------
# LangGraph
# -----------------------------
try:
    from langgraph.graph import StateGraph, END
except Exception:
    StateGraph = None
    END = "__END__"

# -----------------------------
# Tiny TF-IDF RAG utils
# -----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(s or "")]

def _chunk_text(text: str, max_chars: int = 1500, overlap: int = 300) -> List[str]:
    text = text or ""
    if len(text) <= max_chars: return [text]
    chunks: List[str] = []; i = 0
    while i < len(text):
        j = min(len(text), i + max_chars); segment = text[i:j]; k = segment.rfind("\n\n")
        if k > max_chars * 0.5: segment = segment[:k]; j = i + k
        chunks.append(segment)
        if j == len(text): break
        i = max(0, j - overlap)
    return chunks

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

# -----------------------------
# KB ingestion
# -----------------------------
def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _ingest_kb(kb_texts: List[str]) -> List[str]:
    texts: List[str] = []

    # 1) explicit paths
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
            # treat long strings as raw KB text
            if len(entry) > 80: texts.append(entry)

    # 2) env override
    kb_dir_env = os.getenv("INKSPIRE_KB_DIR", "").strip()
    if kb_dir_env:
        p = Path(kb_dir_env)
        if p.exists() and p.is_dir():
            for f in p.rglob("*"):
                if f.suffix.lower() in {".txt", ".md", ".markdown"} and f.is_file():
                    t = _read_text_file(f)
                    if t.strip(): texts.append(t)

    # 3) default folder
    default_dir = Path("kb_folder")
    if default_dir.exists() and default_dir.is_dir():
        for f in default_dir.rglob("*"):
            if f.suffix.lower() in {".txt", ".md", ".markdown"} and f.is_file():
                t = _read_text_file(f)
                if t.strip(): texts.append(t)

    return texts

# -----------------------------
# LLM factory
# -----------------------------
def _get_llm(provider: str):
    provider = (provider or os.getenv("PROVIDER", "openai")).lower()
    if provider == "anthropic" and ChatAnthropic is not None:
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        return ChatAnthropic(model=model, temperature=0.2)
    if ChatOpenAI is None:
        raise RuntimeError("LangChain OpenAI not available. Install langchain-openai and set OPENAI_API_KEY.")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0.2)

# -----------------------------
# Prompts (your full text)
# -----------------------------
SYSTEM_HEADER = """## Role and Purpose

You are an expert in computer science education and disciplinary literacy, specialized in creating reading scaffolds for introductory Python programming textbooks. Your task is to analyze assigned readings and generate instructor annotations that help students read programming texts deeply and meaningfully, following disciplinary literacy principles.

## Theoretical Framework

Your analysis and scaffold generation are grounded in:

1. **Goldman's Disciplinary Literacy Framework** - focusing on CS-specific epistemology (how knowledge is validated), inquiry practices (discipline-specific reasoning strategies), overarching concepts (organizing frameworks), information representation forms (text types and formats), and discourse structures (specialized language patterns)
2. **Hierarchical Skill Development** - understanding that code tracing precedes explanation, which precedes writing
3. **Multimodal Integration** - acknowledging that CS texts require processing code, prose, diagrams, and outputs simultaneously

## Quality Criteria

Ensure scaffolds:

1. **Are Specific** - Target precise sections, not general areas
2. **Build Skills** - Progress from lower to higher cognitive demands
3. **Connect Knowledge** - Link to prior learning and future applications
4. **Promote Discussion** - Include opportunities for peer interaction
5. **Support Transfer** - Help students apply strategies independently
6. **Respect Cognitive Load** - Don't overwhelm with too many scaffolds

## Annotation Tone and Voice

- **Collegial but Guiding** - Position as experienced colleague, not authority
- **Intellectually Curious** - Model genuine engagement with ideas
- **Critically Supportive** - Encourage questioning while providing structure
- **Culturally Humble** - Acknowledge limitations of any single perspective
- **Practically Grounded** - Connect theory to lived educational experiences
""".strip()

PROMPT_A_ANALYZE = """### Phase 1: Material Characterization

When given textbook content, analyze and document:

**A. Content Type Distribution**
- Conceptual explanations (theory, principles, patterns)
- Code examples (syntax demonstrations, complete programs)
- Hybrid sections (code with embedded explanations)
- Visual elements (diagrams, flowcharts, memory models)
- Interactive elements (exercises, practice problems)

**B. Cognitive Load Assessment**
- Vocabulary density (new terms per section)
- Abstraction level (concrete examples vs. abstract concepts)
- Prerequisite knowledge requirements
- Working memory demands (variable tracking, nested structures)

**C. Reading Patterns Required**
- Linear (sequential progression)
- Non-linear (following control flow)
- Reference-oriented (looking up syntax/methods)
- Comparative (multiple examples of same concept)

**D. Disciplinary Features**
- How knowledge is validated (testing, debugging, peer review)
- Problem-solving approaches demonstrated
- Technical discourse markers and conventions
- Representation transitions (prose→code, diagram→implementation)

Return STRICT JSON with keys:
{
  "content_distribution": {"conceptual": "%","code": "%","hybrid": "%","visuals": "%","interactive": "%"},
  "key_modalities": ["..."],
  "cognitive_demands": ["..."],
  "reading_pattern": ["linear"|"non-linear"|"reference-oriented"|"comparative", "..."],
  "notes": ["...brief bullets..."]
}
""".strip()

PROMPT_B_EXTRACT = """### Phase 2: Scaffold-Worthy Area Identification

Identify 5-8 key areas per reading that would benefit most from scaffolding, prioritizing:

**High-Priority Targets:**
1. **Conceptual Bridges** - Where abstract concepts connect to concrete implementations
2. **First Encounters** - Introduction of new programming constructs or patterns
3. **Cognitive Bottlenecks** - Complex code requiring multiple variable tracking
4. **Multimodal Integration Points** - Where students must synthesize across representations
5. **Pattern Recognition Opportunities** - Similar structures across different contexts
6. **Debugging/Testing Scenarios** - Error identification and correction examples
7. **Reflective AI Sections** - Discussions of effective and responsible AI usage in programming

**Selection Criteria:**
- Concepts that build on each other (scaffolding sequences)
- Areas with high transfer potential to future topics
- Common misconception points from CS education research
- Opportunities for both individual and collaborative work

Return STRICT JSON:
{"targets": [
  {
    "anchor_text": "exact quote from the DOCUMENT (<=200 chars)",
    "section_ref": "page or heading if present, else null",
    "content_type": "Conceptual"|"Code"|"Hybrid"|"Problem-Solving",
    "cognitive_focus": "short phrase",
    "intent": "comment"|"question",
    "priority": "High"|"Medium"
  }, ...
]}
Ensure each "anchor_text" is an exact substring of the DOCUMENT.
""".strip()

PROMPT_C_GENERATE = """### Phase 3: Annotation-Based Scaffold Generation

For each identified area, create instructor annotations using these strategies:

**A. For Conceptual Reading Sections:**
- Activation Prompts (connecting to prior knowledge)
- Analogy Construction (relating to familiar concepts)
- Visualization Requests (drawing mental models)
- Explanation Generation (teach-back scenarios)

**B. For Code Reading Sections:**
- PRIMM-Based Scaffolds:
  - Predict: "Before running, what will line 7 output if x=5?"
  - Investigate: "Trace through lines 3-8 using a memory table"
  - Modify: "How would behavior change if we switched lines 4 and 5?"
- Memory Table Construction (systematic variable tracking)
- Control Flow Mapping (execution path visualization)
- Purpose Articulation (explain in plain English without restating code)

**C. For Hybrid Sections:**
- Code-Concept Mapping (link implementation to theory)
- Pattern Extraction (identify reusable structures)
- Comparison Tasks (contrast approaches)
- Translation Exercises (pseudocode ↔ Python)

**D. For Problem-Solving Sections:**
- TIPP&SEE Applications (systematic analysis protocol)
- Metacognitive Prompts (strategy awareness)
- Error Prediction (anticipate common mistakes)
- Test Case Generation (boundary condition thinking)

And the output format:

Part 1: Overview
```markdown
# Reading Scaffold Analysis: [Chapter Title]

## Material Characteristics
- **Content Distribution:** 40% conceptual, 35% code examples, 25% exercises
- **Key Modalities:** Text, code blocks, execution traces, flowcharts
- **Cognitive Demands:** High working memory load in recursion section
- **Reading Pattern:** Primarily non-linear, following function calls

## Identified Focus Areas
1. List comprehension introduction (p. 45-47) - First encounter with Pythonic syntax
2. Recursion visualization (p. 52-54) - High cognitive load, needs memory modeling
3. Exception handling patterns (p. 58-60) - Conceptual to practical bridge
[continue...]
Part 2: each annotation

markdown
Copy code
### Scaffold #N: [Descriptive Title]
**Target Content:** [Page/section reference, or exact quote, or a select area]
**Content Type:** [Conceptual | Code | Hybrid | Problem-Solving]
**Cognitive Focus:** [e.g., variable tracking, pattern recognition, conceptual understanding]

**Scaffold Prompt:**
[The actual question/task/suggestion for students - written in clear, engaging language]

**Strategy Foundation:** [e.g., PRIMM-Investigate, Metacognitive Reflection, Memory Table]
**Expected Engagement Time:** [2-5 minutes | 5-10 minutes | 10-15 minutes]
**Collaboration Mode:** [Individual | Pair | Small Group]

**Learning Objective:** [What students should gain from this scaffold]

**Instructor Note:** [Brief guidance on implementation or common student responses]
---
Output JSON STRICTLY with keys:
{
"overview_markdown": "string",
"items": [
{
"anchor_text": "exact substring from DOCUMENT",
"content_type": "Conceptual"|"Code"|"Hybrid"|"Problem-Solving",
"cognitive_focus": "string",
"differentiation": "Foundational"|"Standard"|"Extension",
"expected_time": "2-5 minutes"|"5-10 minutes"|"10-15 minutes",
"collaboration_mode": "Individual"|"Pair"|"Small Group",
"markdown": "scaffold markdown matching the Part 2 template"
}
]
}
Ensure every anchor_text is an exact substring of DOCUMENT.
""".strip()


@dataclass
class PipelineState:
mode: Literal["annotations","scaffolds"]
full_text: str
kb_texts: List[str]
provider: str = "openai"


# RAG
chunks: List[str] = field(default_factory=list)
kb_chunks: List[str] = field(default_factory=list)
corpus: List[str] = field(default_factory=list)
tfidf_docs: List[Dict[str, float]] = field(default_factory=list)
idf: Dict[str, float] = field(default_factory=dict)

# Agent A/B outputs
analysis_json: Dict[str, Any] = field(default_factory=dict)
targets_json: Dict[str, Any] = field(default_factory=dict)

# Agent C outputs
overview_markdown: str = ""
final_items: List[Dict[str, Any]] = field(default_factory=list)
#-----------------------------
#RAG helpers
#-----------------------------
def _prepare_rag(state: PipelineState) -> None:
    clean = state.full_text.strip()
    state.chunks = _chunk_text(clean, 1500, 300)
    kb_all = _ingest_kb(state.kb_texts)
    kb_chunks: List[str] = []
    for kb in kb_all:
    kb_chunks.extend(_chunk_text(kb, 1000, 100))
    state.kb_chunks = kb_chunks[:300]
    state.corpus = state.chunks + state.kb_chunks
    if not state.corpus: state.corpus = [clean] if clean else [""]
    state.tfidf_docs, state.idf = _build_tfidf_index(state.corpus)

def _nearest_kb(anchor: str, state: PipelineState) -> str:
    idxs = _retrieve(anchor, state.corpus, state.tfidf_docs, state.idf, k=6)
    kb_offset = len(state.chunks)
    kb_indices = [i for i in idxs if i >= kb_offset]
    if kb_indices:
    return state.corpus[kb_indices[0]]
    
return state.corpus[idxs[0]] if idxs else ""

#-----------------------------
#Nodes (Agent A, B, C)
#-----------------------------
def node_analyze(state: PipelineState) -> PipelineState:
    _prepare_rag(state)
    if ChatPromptTemplate is None:
    # If LC/LLM missing, return a minimal stub so downstream can proceed
    state.analysis_json = {
    "content_distribution": {"conceptual":"-", "code":"-", "hybrid":"-", "visuals":"-", "interactive":"-"},
    "key_modalities": [], "cognitive_demands": [], "reading_pattern": [], "notes": ["LangChain not available"]
    }
    return state


    llm = _get_llm(state.provider)
    doc_sample = state.full_text[:4000]
    kb_hint = "\n\n--- KB HINT ---\n" + "\n\n".join(state.kb_chunks[:3]) if state.kb_chunks else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_HEADER + "\n\n" + PROMPT_A_ANALYZE),
        ("user", f"DOCUMENT:\n{doc_sample}{kb_hint}\n\nReturn STRICT JSON only.")
    ])
    resp = llm.invoke(prompt.format_messages())
    try:
        state.analysis_json = json.loads(resp.content)
    except Exception:
        state.analysis_json = {"notes": ["(parse error)"], "raw": resp.content}
    return state
    def node_extract(state: PipelineState) -> PipelineState:
    if ChatPromptTemplate is None:
    state.targets_json = {"targets": []}
    return state

llm = _get_llm(state.provider)
doc_sample = state.full_text[:8000]
analysis = json.dumps(state.analysis_json, ensure_ascii=False)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_HEADER + "\n\n" + PROMPT_B_EXTRACT),
    ("user", f"DOCUMENT:\n{doc_sample}\n\nANALYSIS_JSON:\n{analysis}\n\nReturn STRICT JSON only.")
])
resp = llm.invoke(prompt.format_messages())
try:
    parsed = json.loads(resp.content)
except Exception:
    parsed = {"targets": []}

# filter: ensure quotes are actual substrings
items = []
for t in parsed.get("targets", []):
    q = (t.get("anchor_text") or "").strip()
    if q and q in state.full_text:
        items.append(t)
parsed["targets"] = items[:8]
state.targets_json = parsed
return state
def node_generate(state: PipelineState) -> PipelineState:
if ChatPromptTemplate is None:
state.final_items = []
state.overview_markdown = ""
return state

swift
Copy code
llm = _get_llm(state.provider)
doc_sample = state.full_text[:12000]
targets = json.dumps(state.targets_json, ensure_ascii=False)

# Build KB context per target
kb_contexts = []
for t in state.targets_json.get("targets", []):
    q = (t.get("anchor_text") or "").strip()
    if not q: continue
    kb_hit = _nearest_kb(q, state)
    if kb_hit:
        kb_contexts.append({"anchor_text": q, "kb_snippet": kb_hit[:900]})
kb_json = json.dumps({"contexts": kb_contexts}, ensure_ascii=False)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_HEADER + "\n\n" + PROMPT_C_GENERATE),
    ("user", f"DOCUMENT:\n{doc_sample}\n\nTARGETS_JSON:\n{targets}\n\nKB_CONTEXTS:\n{kb_json}\n\nReturn STRICT JSON with keys overview_markdown and items.")
])
resp = llm.invoke(prompt.format_messages())
try:
    parsed = json.loads(resp.content)
except Exception:
    parsed = {"overview_markdown": "", "items": []}

# final exact-substring filter
final_items = []
for it in parsed.get("items", []):
    a = (it.get("anchor_text") or "").strip()
    if a and a in state.full_text:
        final_items.append(it)

state.final_items = final_items[:8]
state.overview_markdown = parsed.get("overview_markdown", "")
return state
-----------------------------
Graph builder
-----------------------------
def _compile_graph():
if StateGraph is None: return None
graph = StateGraph(PipelineState)
graph.add_node("AgentA_Analyze", node_analyze)
graph.add_node("AgentB_Extract", node_extract)
graph.add_node("AgentC_Generate", node_generate)
graph.set_entry_point("AgentA_Analyze")
graph.add_edge("AgentA_Analyze", "AgentB_Extract")
graph.add_edge("AgentB_Extract", "AgentC_Generate")
graph.add_edge("AgentC_Generate", END)
return graph.compile()

-----------------------------
Public APIs
-----------------------------
def _run(full_text: str, kb_texts: List[str], provider: str):
st = PipelineState(mode="annotations", full_text=full_text or "", kb_texts=kb_texts or [], provider=provider or "openai")
app = _compile_graph()
if app is None:
return "", []
res: PipelineState = app.invoke(st) # type: ignore
return res.overview_markdown, res.final_items

def generate_full_report(full_text: str, kb_texts: List[str], provider: str = "openai", out_dir: Optional[str] = None) -> Dict[str, Any]:
overview_md, items = _run(full_text, kb_texts, provider)
result = {"overview_markdown": overview_md or "", "items": items or []}
if out_dir:
outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
(outp / "overview.md").write_text(overview_md or "", encoding="utf-8")
(outp / "annotations.json").write_text(json.dumps(items or [], ensure_ascii=False, indent=2), encoding="utf-8")
result["overview_path"] = str(outp / "overview.md")
result["annotations_path"] = str(outp / "annotations.json")
return result

Back-compat: returns Part 2-like items (markdown in "text")
def generate_annotations(full_text: str, kb_texts: List[str], provider: str = "openai") -> List[Dict]:
_, items = _run(full_text, kb_texts, provider)
out = []
for it in items:
out.append({
"anchor_text": it.get("anchor_text",""),
"type": it.get("type","question"), # may be absent; preserved for pipeline compatibility
"text": it.get("markdown","").strip() or it.get("text",""),
"content_type": it.get("content_type","Conceptual"),
"cognitive_focus": it.get("cognitive_focus","conceptual understanding"),
"differentiation": it.get("differentiation","Standard"),
"expected_time": it.get("expected_time","5-10 minutes"),
"collaboration_mode": it.get("collaboration_mode","Individual"),
})
return out[:8]

Alias: return scaffold dicts as-is
def generate_scaffolds(full_text: str, kb_texts: List[str], provider: str = "openai") -> List[Dict]:
_, items = _run(full_text, kb_texts, provider)
return items[:8]