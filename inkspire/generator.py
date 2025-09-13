from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
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

_BODY_CUES = re.compile(r"\b(abstract|introduction)\b", flags=re.I)

def _extract_body(text: str) -> str:
    """Return the substring starting from 'Abstract' or 'Introduction' if present."""
    if not text:
        return ""
    m = _BODY_CUES.search(text)
    return text[m.start():] if m else text


# ---------------------------- State ----------------------------

@dataclass
class PipelineState:
    material_text: str
    kb_texts: List[str] = field(default_factory=list)
    provider: str = "gemini"
    # Outputs
    material_summary: str = ""
    candidate_spans: List[str] = field(default_factory=list)
    scaffolds: List[Dict[str, Any]] = field(default_factory=list)
    agent_a: Dict[str, Any] = field(default_factory=dict)
    agent_a_path: Optional[str] = None
    # User-supplied human prompt
    user_human_prompt: str = ""



# ---------------------------- File I/O helpers ----------------------------
def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        _dbg(f"[FileIO] Read {len(data)} chars from {path}")
        return data
    except Exception as e:
        _dbg(f"[FileIO] Failed to read {path}: {e}")
        return ""

def _write_text_file(path: str, content: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        _dbg(f"[FileIO] Wrote {len(content)} chars to {path}")
    except Exception as e:
        _dbg(f"[FileIO] Failed to write {path}: {e}")



# ---------------------------- Prompt loader ----------------------------

def _load_user_prompts() -> str:
    """Load and join all .txt prompts from inkspire/prompts/."""
    # Resolve .../inkspire/prompts relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_dir = os.path.join(base_dir, "prompts")

    # Fallback: try CWD/inkspire/prompts if not found (e.g., editable runs)
    if not os.path.isdir(prompt_dir):
        alt = os.path.join(os.getcwd(), "inkspire", "prompts")
        if os.path.isdir(alt):
            prompt_dir = alt
        else:
            _dbg(f"[Prompts] Prompt dir not found: {prompt_dir} and {alt}")
            return ""

    chunks: List[str] = []
    for fname in sorted(os.listdir(prompt_dir)):
        path = os.path.join(prompt_dir, fname)
        if os.path.isfile(path) and fname.lower().endswith(".txt"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                if txt:
                    chunks.append(txt)
            except Exception as e:
                _dbg(f"[Prompts] Failed to read {path}: {e}")

    joined = "\n\n---\n\n".join(chunks)
    # Cap to avoid blowing context; adjust if you have larger budgets.
    capped = joined[:4000]
    _dbg(f"[Prompts] Loaded {len(chunks)} files; total chars={len(capped)}")
    return capped



# --------------------- Gemini-backed agents (LangChain) ---------------------

def _agent_a_llm(llm: "ChatGoogleGenerativeAI"):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Agent A."),
        ("human",
         # User-supplied instructions first (if any), then your fixed spec, then data.
         "USER INSTRUCTIONS (verbatim):\n{user_human_prompt}\n\n"
         "ROLE: Agent A (Material Characterization).\n\n"
         "When given textbook content (MATERIAL), analyze and document:\n"
         "A. Content Type Distribution\n"
         "- Conceptual explanations (theory, principles, patterns)\n"
         "- Code examples (syntax demonstrations, complete programs)\n"
         "- Hybrid sections (code with embedded explanations)\n"
         "- Visual elements (diagrams, flowcharts, memory models)\n"
         "- Interactive elements (exercises, practice problems)\n\n"
         "B. Cognitive Load Assessment\n"
         "- Vocabulary density (new terms per section)\n"
         "- Abstraction level (concrete examples vs. abstract concepts)\n"
         "- Prerequisite knowledge requirements\n"
         "- Working memory demands (variable tracking, nested structures)\n\n"
         "C. Reading Patterns Required\n"
         "- Linear, Non-linear, Reference-oriented, Comparative\n\n"
         "D. Disciplinary Features\n"
         "- Validation (testing/debugging/peer review), problem-solving approaches,\n"
         "  discourse markers, representation transitions (prose→code, diagram→impl)\n\n"
         "OUTPUT (Markdown only; no code fences, no JSON):\n"
         "# Reading Scaffold Analysis: [Chapter Title]\n\n"
         "## Material Characteristics\n"
         "- **Content Distribution:** <e.g., 40% conceptual, 35% code examples, 25% exercises>\n"
         "- **Key Modalities:** <e.g., Text, code blocks, execution traces, flowcharts>\n"
         "- **Cognitive Demands:** <e.g., High working memory load in recursion section>\n"
         "- **Reading Pattern:** <e.g., Primarily non-linear, following function calls>\n\n"
         "RULES: Output valid Markdown only (no code fences), ≤300 words, infer chapter title if absent.\n\n"
         "MATERIAL:\n{material_text}")
    ])
    chain = prompt | llm | StrOutputParser()

    def _run(state: PipelineState) -> PipelineState:
        raw_md = chain.invoke({
            "material_text": state.material_text[:12000],
            "user_human_prompt": state.user_human_prompt
        })
        md_output = raw_md.strip()
        payload = {"markdown": md_output}
        state.material_summary = md_output

        path = os.getenv("INKSPIRE_AGENT_A_PATH", "agent_a.json")
        try:
            _write_text_file(path, json.dumps(payload, ensure_ascii=False, indent=2))
            state.agent_a_path = path
            state.agent_a = payload
        except Exception as e:
            _dbg(f"[AgentA] write failed: {e}")
            state.agent_a_path = None
            state.agent_a = payload

        return state

    return RunnableLambda(_run)




def _agent_b_llm(llm: "ChatGoogleGenerativeAI"):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Agent B."),
        ("human",
         "USER INSTRUCTIONS (verbatim):\n{user_human_prompt}\n\n"
         "ROLE: Agent B (Scaffold-Worthy Passage Identification).\n\n"
         "Identify 5–8 key areas per reading that would benefit most from scaffolding, prioritizing:\n\n"
         "**High-Priority Targets:**\n"
         "1. Conceptual Bridges – Where abstract concepts connect to concrete implementations\n"
         "2. First Encounters – Introduction of new programming constructs or patterns\n"
         "3. Cognitive Bottlenecks – Complex code requiring multiple variable tracking\n"
         "4. Multimodal Integration Points – Where students must synthesize across representations\n"
         "5. Pattern Recognition Opportunities – Similar structures across different contexts\n"
         "6. Debugging/Testing Scenarios – Error identification and correction examples\n"
         "7. Reflective AI Sections – Discussions of effective and responsible AI usage in programming\n\n"
         "**Selection Criteria:**\n"
         "- Concepts that build on each other (scaffolding sequences)\n"
         "- Areas with high transfer potential to future topics\n"
         "- Common misconception points from CS education research\n"
         "- Opportunities for both individual and collaborative work\n\n"
         "CONSTRAINTS:\n"
         "- Passages MUST be exact substrings copied verbatim from MATERIAL.\n"
         "- Each passage length: 70–350 chars.\n"
         "- Exclude acknowledgments, author lists, affiliations, emails, references.\n\n"
         "ADDITIONAL CONTEXT (Agent A Markdown):\n{agent_a_md}\n\n"
         "MATERIAL:\n{material_text}\n\n"
         "OUTPUT: JSON array of strings, each an exact substring.")
    ])
    chain = prompt | llm | StrOutputParser()

    def _run(state: PipelineState) -> PipelineState:
        src = state.material_text or ""
        a_md = state.agent_a.get("markdown", "")

        raw_output = chain.invoke({
            "material_text": src[:40000],
            "agent_a_md": a_md[:4000],
            "user_human_prompt": state.user_human_prompt
        })
        # ... (rest of your parsing/mapping/dedup logic unchanged)
        spans: List[str] = []
        try:
            data = json.loads(raw_output)
            if isinstance(data, list):
                spans = [s for s in data if isinstance(s, str)]
        except Exception:
            spans = re.findall(r'"([^"]{40,800})"', raw_output)[:8]

        spans = [s.strip() for s in spans if s and not _looks_like_ack(s)]

        mapped: List[str] = []
        for s in spans:
            rs, end_idx = find_first_occurrence(src, s)
            if rs != -1:
                exact = src[rs:end_idx]
                mapped.append(_trim_span_to_reasonable_window(src, exact))
            else:
                mapped.append(s)

        seen, out = set(), []
        for m in mapped:
            key = re.sub(r"\s+", " ", m.lower())[:200]
            if key in seen:
                continue
            seen.add(key)
            out.append(m)

        state.candidate_spans = out[:8]
        _dbg(f"[AgentB] candidates={len(state.candidate_spans)}")
        return state

    return RunnableLambda(_run)



def _agent_c_llm(llm: "ChatGoogleGenerativeAI"):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Agent C."),
        ("human",
         "USER INSTRUCTIONS (verbatim):\n{user_human_prompt}\n\n"
         "For each identified area, create instructor annotations using these strategies:\n\n"
         "**A. For Conceptual Reading Sections:**\n"
         "- Activation Prompts (connecting to prior knowledge)\n"
         "- Analogy Construction (relating to familiar concepts)\n"
         "- Visualization Requests (drawing mental models)\n"
         "- Explanation Generation (teach-back scenarios)\n\n"
         "**B. For Code Reading Sections:**\n"
         "- PRIMM-Based Scaffolds:\n"
         "  - Predict: \"Before running, what will line 7 output if x=5?\"\n"
         "  - Investigate: \"Trace through lines 3–8 using a memory table\"\n"
         "  - Modify: \"How would behavior change if we switched lines 4 and 5?\"\n"
         "- Memory Table Construction (systematic variable tracking)\n"
         "- Control Flow Mapping (execution path visualization)\n"
         "- Purpose Articulation (explain in plain English without restating code)\n\n"
         "**C. For Hybrid Sections:**\n"
         "- Code-Concept Mapping (link implementation to theory)\n"
         "- Pattern Extraction (identify reusable structures)\n"
         "- Comparison Tasks (contrast approaches)\n"
         "- Translation Exercises (pseudocode ↔ Python)\n\n"
         "**D. For Problem-Solving Sections:**\n"
         "- TIPP&SEE Applications (systematic analysis protocol)\n"
         "- Metacognitive Prompts (strategy awareness)\n"
         "- Error Prediction (anticipate common mistakes)\n"
         "- Test Case Generation (boundary condition thinking)\n\n"
         "## Output Format\n\n"
         "For each scaffold, provide:\n\n"
         "### Scaffold #N: [Descriptive Title]\n"
         "**Content Type:** [Conceptual | Code | Hybrid | Problem-Solving]\n"
         "**Cognitive Focus:** [e.g., variable tracking, pattern recognition, conceptual understanding]\n\n"
         "**Scaffold Prompt:**\n"
         "[The actual question/task/suggestion for students - written in clear, engaging language]\n\n"
         "**Strategy Foundation:** [e.g., PRIMM-Investigate, Metacognitive Reflection, Memory Table]\n"
         "**Expected Engagement Time:** [2-5 minutes | 5-10 minutes | 10-15 minutes]\n"
         "**Collaboration Mode:** [Individual | Pair | Small Group]\n\n"
         "**Learning Objective:** [What students should gain from this scaffold]\n\n"
         "**Instructor Note:** [Brief guidance on implementation or common student responses]\n"
         "---\n\n"
         "TARGET SPANS (JSON):\n{spans_json}\n\n"
         "GLOBAL CONTEXT (truncated):\n{material_head}\n\n"
         "OUTPUT: return ONLY a JSON array of objects with keys \"target_span\" and \"text\".\n"
         "Max 180 words per scaffold. No code fences; no extra prose.")
    ])
    chain = prompt | llm | StrOutputParser()


    def _run(state: PipelineState) -> PipelineState:
        spans = state.candidate_spans[:8]
        src = state.material_text or ""

        raw_output = chain.invoke({
            "spans_json": json.dumps(spans, ensure_ascii=False),
            "material_head": src[:1200],
            "user_human_prompt": state.user_human_prompt
        })

        out: List[Dict[str, Any]] = []
        parsed: List[Dict[str, Any]] = []
        try:
            data = json.loads(raw_output)
            if isinstance(data, list):
                parsed = [d for d in data if isinstance(d, dict)]
        except Exception:
            m = re.search(r'\[[\s\S]*\]', raw_output)
            if m:
                try:
                    data = json.loads(m.group(0))
                    if isinstance(data, list):
                        parsed = [d for d in data if isinstance(d, dict)]
                except Exception:
                    parsed = []

        for idx, item in enumerate(parsed, start=1):
            span = str(item.get("target_span", "")).strip()
            txt = str(item.get("text", "") or item.get("markdown", "")).strip()
            if not span or not txt:
                continue
            rs, end_idx = find_first_occurrence(src, span)
            if rs != -1:
                exact = src[rs:end_idx]
                windowed = _trim_span_to_reasonable_window(src, exact)
                anchor = {"start": rs, "end": end_idx, "fragment": exact}
            else:
                windowed = span
                anchor = {"start": -1, "end": -1, "fragment": ""}

            if not txt.lstrip().startswith("### Scaffold #"):
                txt = f"### Scaffold #{idx}: (Auto)\n" + txt

            out.append({
                "title": f"Scaffold #{idx}",
                "target_span": windowed,
                "text": txt,
                "anchor": anchor,
            })

        state.scaffolds = out
        return state

    return RunnableLambda(_run)



# --------------------------- Orchestration ---------------------------

def _build_gemini(model_name: str = "gemini-2.5-flash", temperature: float = 0.2):
    """
    Construct the Gemini chat model via langchain_google_genai.
    Raises clear errors if LangChain or GOOGLE_API_KEY are not available.
    """
    if not _LC_AVAILABLE:
        raise RuntimeError("LangChain / langchain-google-genai not available (install or check imports).")
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Export it before running.")
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)



def _run_llm_pipeline(state: PipelineState, model_name: str = "gemini-2.5-flash") -> PipelineState:
    llm = _build_gemini(model_name=model_name, temperature=0.2)
    a = _agent_a_llm(llm)
    b = _agent_b_llm(llm)
    c = _agent_c_llm(llm)
    s = a.invoke(state)
    s = b.invoke(s)
    s = c.invoke(s)
    return s



# ---------------------------- Public API ----------------------------

def generate_annotations(material_text: str, kb_texts: List[str], provider: str = "gemini") -> List[Dict[str, Any]]:
    state = PipelineState(
        material_text=material_text or "",
        kb_texts=kb_texts or [],
        provider=(provider or "gemini").lower(),
    )

    # Always load prompts from inkspire/prompts/
    state.user_human_prompt = _load_user_prompts()

    if len(state.material_text.strip()) < 40:
        return []

    state = _run_llm_pipeline(state, model_name="gemini-2.5-flash")
    return state.scaffolds[:8]