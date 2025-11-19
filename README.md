# Scope

Scope is a research interface for reading language-model behavior through multiple complementary
lenses—token probabilities, counterfactual generations, model comparison, and Logit Lens views—
all in one place. Instead of juggling ad-hoc scripts and chat transcripts, Scope makes token-level
signals the default unit of analysis and lets researchers pivot into deeper workflows in seconds.

---

## Abstract
Scope accelerates empirical AI-safety workflows by unifying token-level probability inspection,
counterfactual editing, model comparison, and mechanistic interpretability handoffs inside a single
chat-style UI. The tool streams model generations, records per-token probabilities, ranks, and top
alternatives, and overlays this substrate with lenses for forcing tokens, branching hypotheses,
diffing models, and launching Logit Lens visualizations. We argue that token probabilities are the
natural starting point for many safety investigations—hallucination detection, refusal analysis,
jailbreaking, calibration, decoding—and Scope compresses hours of manual probing into minutes of
interactive inspection.

---

## Motivation

Current workflows oscillate between:

1. **High-level evaluations** (benchmarks, red-team suites) that flag failures but rarely show why.
2. **Mechanistic interpretability tools** that explain circuits but are too heavyweight for every
   prompt.

Scope fills the “missing middle.” It treats tokens and their probabilities as first-class objects,
giving researchers a fast, shared workspace to triage model behavior before escalating to heavier
tooling. Success means jailbreak studies, refusal audits, calibration checks, and hallucination
investigations routinely start in Scope, making probability inspection the default—not an optional
API flag.

---

## Methodology Snapshot

1. **Surveyed safety research** (jailbreaking, RL, hallucinations, calibration, decoding) to catalog
   how teams already inspect log probs, entropy spikes, or token rankings.
2. **Extracted concrete workflows** (prefilling refusal tokens, comparing checkpoints, watching
   divergence between models) and mapped them to minimal UI affordances.
3. **Specified phased delivery** to keep the interface modular:
   - Phase 1: Chat UI, streaming tokens, hover tooltips, Logit Lens integration. ✅
   - Phase 2: Investigation workflows (prefill, Diff view, export, annotations). ⏳
   - Phase 3: Advanced lenses (hallucination alerts, calibration, decoding modes).
   - Phase 4: Mechanistic integrations (TransformerLens / NDIF hand-offs).
4. **Iterative design** with researchers to ensure features match real safety tasks.

---

## Prototype Highlights

- **TokenLens** (default): color-coded tokens, entropy-aware tooltips, vocabulary search, injection/
  branching, Logit Lens modal (heatmap + Chart.js line chart).
- **Force Assistant Start** (“BreakLens” capability): prefill responses, highlight injected tokens,
  maintain dashed borders for prefills even after generation completes.
- **DiffLens**: run an analysis model over the same transcript to expose probability gaps and implicit
  disagreements.

All modes share the same probability substrate, showing how a single stream of token stats powers
multiple investigative lenses.

---

## Challenges & Open Questions

- **Model coverage**: log-probabilities are inconsistently exposed across APIs; we lean on providers
  like OpenRouter or local inference where possible.
- **Tokenizer alignment**: probability diffs require shared tokenization; richer alignment strategies
  remain future work.
- **Mode scoping**: deciding which specialized lenses truly help researchers requires iterative
  testing and field feedback.

---

## Roadmap Overview

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Core chat UI, streaming tokens, hover tooltips, Logit Lens | ✅ |
| 2 | Investigation workflows (prefill, diff, export, annotations) | In progress |
| 3 | Specialized modes (hallucination, calibration, decoding) | Planned |
| 4 | Mechanistic handoffs (steering vectors, activation patching) | Planned |

---

## Running Locally

```bash
git clone <repo-url>
cd TokenLens
python -m venv .venv
.venv\Scripts\activate      # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000` and start chatting. Double-click any token or use the tooltip actions
to inject alternatives, branch generations, or open Logit Lens.

---

## Repository Layout

- `app.py` – Flask backend, SSE streaming, DiffLens / Logit Lens endpoints.
- `templates/chat.html` – Main interface (token rendering, modals, search, charts).
- `static/chat.css` – Styling for the chat, tool sidebar, and Logit Lens modal.
- `probability_monitor.py`, `prompts/`, `probabilities/` – CLI helpers and canned datasets.
- `templates/index.html`, `static/style.css` – Legacy interface for archived analyses.

---

## Contributing

1. Branch from `main`.
2. Scope front-end changes to `templates/` + `static/`; keep backend updates isolated in `app.py`.
3. Include tests or repro steps for backend changes; add screenshots/gifs for UI tweaks.
4. Open a PR—feedback, feature ideas, and bug reports are welcome.

Scope aims to become the standard cockpit for empirical alignment work.
If you have a workflow that relies on token-level signals, reach out so we can build it in. 

