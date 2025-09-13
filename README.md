# Inkspire – Perusall Scaffolding Generator

Inkspire is a tool that reads course materials (PDF/text), runs a multi-agent LLM pipeline (Agent A, B, C), and generates **instructional scaffolds** that can be posted to [Perusall](https://perusall.com/) as annotations.

---

## 🚀 How to Run

### 1. Setup
Clone the repo (or copy the files), create a virtual environment, and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Set up environment variables (create a `.env` file in the project root):

```bash
PERUSALL_BASE_URL=https://app.perusall.com/api/v1
PERUSALL_API_TOKEN=your_X_api_token
PERUSALL_INSTITUTION=you_X_institution

LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here
```

### 2. Add Human Prompts
Put all your user-facing instructions inside **`inkspire/prompts/`** as `.txt` files, e.g.:

- `course.txt` (course-level prompts)
- `assignment.txt` (assignment-level prompts)

These are automatically concatenated and injected into **all three agents** (A, B, C).

### 3. Run a Dry Run

First, you need to get **`<your-course-id>`**, **`<your-document-id>`**, **`<your-assignment-id>`**, and **`<your-user-id>`** from the [Perusall API Documentation](https://apidocs.perusall.com/#introduction).

To preview scaffold annotations without posting:

```bash
INKSPIRE_DEBUG=1 inkspire \
  --course-id <your-course-id> \
  --document-id <your-document-id> \
  --dry-run \
  --out preview.json
```

This will generate `preview.json` containing all candidate scaffolds.

### 4. Post to Perusall
Remove the `--dry-run` flag:

```bash
inkspire \
  --course-id <your-course-id>\
  --document-id <your-document-id> \
  --assignment-id <your-assignment-id> \
  --user-id <your-user-id> \
  --out posted_receipts.json
```

Successful posts are tracked in `posted_receipts.json`.

---

## 📂 Project Structure

```
inkspire/
│
├── prompts/                # All human prompt files (editable by instructors)
│   ├── course.txt
│   └── assignment.txt
│
├── cli.py                  # CLI entry point (`inkspire` command)
├── generator.py            # Main multi-agent pipeline (Agent A, B, C)
├── locator.py              # Anchor text locator (maps spans → Perusall ranges)
├── pipeline.py             # Orchestrates cleaning, generation, posting
├── perusall_api.py         # Wrapper around Perusall API (POST, receipts, etc.)
├── cleaner.py              # Prepares and normalizes raw document text
├── config.py               # Configuration helpers
└── __init__.py
```

**Other files in root:**
- `.env` → Your API keys and secrets
- `pyproject.toml` → Dependencies and build config
- `.gitignore` → Ignore unnecessary files in version control

**Generated during runs:**
- `preview.json` → Dry-run scaffold preview  
- `agent_a.json` → Agent A’s output (Markdown summary)  
- `posted_receipts.json` → Records successfully posted comments  
- `skipped_annotations.json` → Logs skipped/failed scaffolds  
- `inkspire.egg-info/` → Auto-generated when installed locally  

---

## 🧩 Agents Overview

- **Agent A (Material Characterization)**  
  Reads raw material and produces a **Markdown analysis** of content types, cognitive load, and reading patterns.  

- **Agent B (Passage Identification)**  
  Picks **5–8 verbatim spans** from material that are strong candidates for scaffolding.  

- **Agent C (Scaffold Generator)**  
  Expands each span into a **full instructional scaffold** (with prompts, strategies, objectives, and instructor notes).  

---

## 🧹 Cleaning Debug/Cache Files

After experiments, you can clean up debug output with:

```bash
rm -f agent_a.json preview.json posted_receipts.json skipped_annotations.json
rm -rf inkspire.egg-info
```

---

## ✅ Example Workflow

1. Place your prompts in `inkspire/prompts/`
2. Run `inkspire --dry-run` to generate `preview.json`
3. Inspect scaffolds → tweak prompts if needed
4. Run without `--dry-run` to post directly to Perusall

---

## 📖 API Reference

- Perusall API docs: https://apidocs.perusall.com/  
- Inkspire uses `POST /api/v1/courses/{course_id}/documents/{document_id}/comments` for scaffolds.

---

Happy scaffolding 🎓✨
