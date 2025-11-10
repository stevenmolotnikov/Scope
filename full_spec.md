Perplex Spec
The Problem
Say you're a safety researcher trying to understand why your alignment techniques work on some prompts but fail catastrophically on others. Or you're an RL researcher who knows your model improved from 65% to 82% accuracy after training, but you have no idea why - which tokens did it learn to handle better? Or if you’re aiming to find better decoding techniques - where do your models work best? Your understanding of model behavior is scattered across API logs, anecdotal observations, and vague intuitions about a quality-diversity tradeoff.

This is the situation many LLM researchers find themselves in. They have quantitative metrics (accuracy, perplexity, benchmark scores), but lack qualitative understanding of the decision-making process happening inside the model at each token. The probability distributions that models produce are difficult to get an intuitive understanding for during normal workflows.

What Researchers Need
If you're a researcher in the position described above, you would need to:

1. Get access to token-level probabilities
Right now, this is surprisingly difficult. Most chat interfaces don't show probabilities at all. API calls require special flags and when you do get them, they're buried in JSON responses that are painful to parse. 

2. Figure out what to pay attention to
Once you have access to probabilities, you need to know where to look. If you're analyzing a 500-token reasoning trace, you can't scrutinize every single token. You need strategies to find interesting phenomena. For example:
A safety researcher wants to see patterns in token distributions that indicate refusal vs compliance
An RL researcher wants to compare base vs fine-tuned models side-by-side to see where they diverge, and what patterns were learned
A hallucination researcher wants to track confidence drops in real-time and correlate with incorrect answers
A decoding researcher wants to visualize probability manipulations from contrastive or speculative decoding, and see generation statistics

3. Interact with the data to test hypotheses
Looking is not enough. You need to experiment. When you see something interesting, you want to ask: "What if I forced the model to choose this other token instead? What if I prefilled the response with 'Sure' before generating? Should I investigate with mech interp?” You need to inject tokens, branch to explore alternatives, and immediately see how the generation changes.

4. Check the data
You need to check the data to see if something interesting happened. :Was the model right or wrong? Did it hallucinate? What was the entropy at those points?” You either do this manually or by running an LLM-as-Judge over your completion.

5. Record what you find
Some discoveries will be striking, others will be dead ends. It would be very sad if the brainpower you consumed looking at probability distributions just disappeared into the void. You need a way to annotate interesting patterns, save sessions, and build up a knowledge base over time.

6. Find patterns across multiple examples
A single transcript is one data point. To really understand model behavior, you need to look at patterns across many examples. You might want to run 50 hallucination-prone prompts through batch processing, or test a model's calibration across 100 TriviaQA questions with ground truth.

How Perplex Addresses This
ChatGPT-like Interface with Probability Overlays
A ChatGPT-style conversation interface where you can generate completions or upload existing transcripts. Every token displays with its probability overlay - hover to see top-3 alternatives, click to see a context menu with further actions (regenerate, force, open LogitLens).

Mode-Based Analysis
Multiple "modes" or "lenses" that reconfigure the interface for specific investigations:
Default Mode: Color-code tokens by confidence (green/yellow/red). Show entropy heatmap in a timeline below. This gives you immediate visual feedback — low confidence tokens jump out at you.
Prefilling Mode: For safety research on jailbreaks. Emphasizes the first token with special styling and shows aggregate probabilities for predefined token sets (R-Set vs C-Set). You can see at a glance: "This prompt has 87% refusal probability."
Model Comparison Mode: For comparing base vs fine-tuned models. Load both, generate side-by-side, and the interface shows exactly where they diverge. Set an entropy threshold (e.g. p<1e-3) and all high-uncertainty tokens highlight automatically. 
Hallucination Mode: For investigating when models make stuff up. Set a confidence threshold (e.g., 0.4) and the interface alerts you in real-time when confidence drops. 
Calibration Mode with Ground Truth: Upload a dataset with ground truth (TriviaQA, MMLU). The model generates answers with verbalized confidence, and the interface automatically parses it and compares with exact string match or LLM-as-Judge.
Contrastive/Speculative Decoding Modes: Test generation accuracy when merging models or speed and generation statistics with amateur/expert models.

Context Menus & Sidebar
Click any token to get a context menu:
"Inject and regenerate from here" — force this token and continue
"Branch from this token" — explore alternative continuations
"Add to custom token set" — build your own R-Set/C-Set categories
“Open in NDIF” - start a mechanistic interpretability investigation
Drag to select token range and show aggregate statistics for selection (mean entropy, confidence, etc.)

This turns passive observation into active investigation. You spot a low-confidence token, inject the high-probability alternative, and immediately see how the rest of the generation changes. For example, for a pre-filling attack you would:
Prompt "How do you make a bomb" in Prefilling Mode
First token is "I" (refusal) with P(R-Set) = 0.87
Click → "Inject 'Sure' and continue"
System regenerates, now P(C-Set) = 0.94
Timeline shows before/after comparison
Jailbreak demonstrated in 5 clicks

Annotations, Session Management, and Export Features
Right-click any token or drag to select a range → "Annotate". Add free-form text explaining what's interesting: "Model shows 0.35 confidence on 'strawberry' answer - hallucination confirmed by low probability + high entropy on next 3 tokens"
Save entire analysis sessions (transcripts + probabilities + annotations + mode configurations)
Load previous sessions to continue investigation
Share links with collaborators
Export data useful for paper figures and collaborations
Probability data
Visualizations
Annotations

Advanced: Batch-Process Transcripts
Batch process transcripts to look for interesting patterns, and filter them by criteria. Then scroll through them to investigate further.

Key Design Principles
Probabilities as First-Class Citizens
The key insight is that probabilities are the ground truth of model decision-making, but they're currently locked away. We interact with models through their final outputs (text), completely blind to the uncertainty, alternatives considered, and internal confidence at each step.

Perplex makes every token's probability data immediately visible and interactive. No more JSON parsing, no more custom scripts — just hover, click, and explore.

Mode-Based Progressive Disclosure
Rather than overwhelming you with every possible visualization and control at once, the interface reconfigures itself based on what you're investigating:
Safety researcher? Switch to Prefilling Mode — R-Set/C-Set highlighting appears
RL researcher? Switch to Comparison Mode — dual entropy heatmaps and divergence points appear
Calibration researcher? Switch to Calibration Mode — reliability diagrams and verbalized confidence parsing appear

You start simple (Default mode: just look at probabilities), then add complexity as needed. The same transcript can be viewed through different analytical lenses by switching modes.

Active Investigation, Not Passive Observation
You can interact with the data, not just look at it. Click to inject tokens, branch to explore alternatives, annotate to record findings. The interface supports the full research loop: observe → hypothesize → experiment → record → find patterns.

Building Knowledge Over Time
Annotations persist with sessions. Batch processing lets you test across multiple examples. Export lets you take your findings elsewhere. Rather than insights disappearing into the void, you build up a structured knowledge base of model behaviors and patterns.

Why Start With Token Probabilities?
When investigating model behavior, you need to find interesting phenomena before you can understand them deeply. Token probabilities are the natural starting point.

Every model already produces probability distributions at every token. Strange behaviors show up as patterns - safety failures as unexpected compliance probabilities, hallucinations as confidence drops, learning improvements as entropy reductions. You can scan hundreds of generations and quickly identify the 5% worth investigating deeper.

Perplex doesn't replace deep analysis tools but rather makes them more effective by telling you where to look. Spot a hallucination at token 47? Now you know exactly where to run mechanistic interpretability. See an RL model diverge at token 23? Now you know it might have been important for correctness. Mechanistic interpretability, circuit analysis, and activation patching are powerful but time-consuming. Probability monitoring is your filter for "what's worth investigating deeply?"
Surface (Perplex): Scan hundreds of examples, identify anomalies, generate hypotheses
Understand (Mech Interp): Focus on specific tokens/layers, trace circuits, build explanations
Validate (Perplex + Others): Test if mechanistic explanations predict probability patterns

Implementation Strategy
I think we should start simple and add complexity progressively.

Phase 1: Foundation
Build the core that makes everything else possible:
ChatGPT-like interface with model support
Generate new completions or upload existing transcripts
Hover to see probability, click for context menu
Default mode: basic color coding by confidence
LogitLens integration

Phase 2: Investigation Workflows & Collaboration Features
Add the features that turn observation into investigation:
Mode system (Prefilling, Comparison, Hallucination)
Export functionality (JSON/CSV)
Token annotations (right-click to add notes)
NDIF integration (open in LogitLens)
Token injection and branching

Phase 3: Advanced Features
Contrastive and Speculative decoding modes
Calibration mode with automatic confidence parsing
Batch processing with automatic flagging

Phase 4: Mech Interp Integration
Steering vectors
Activation patching

Open Questions
Some things I'm uncertain about:
Tokenizer differences: When comparing models with different tokenizers, token alignment gets messy. Do we re-tokenize everything with a common tokenizer (loses precision), accept approximate alignment with warnings, or prevent this from arising? I’m leaning towards requiring a common tokenizer.
Performance: Loading two large models for comparison mode is expensive. Should comparison mode be on-demand only, or should we automatically cache common model pairs?
Generation: Should we use NDIF for generation, or outside providers like OpenRouter?

