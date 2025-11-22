# Scope

Scope is a research interface for reading language-model behavior through multiple complementary lenses—token probabilities, counterfactual generations, model comparison, and Logit Lens visualizations—all in one interactive chat interface.

---

## Features

### 🔍 **Token View (TokenLens)**
- **Real-time probability visualization**: Every generated token is color-coded by probability or rank
- **Interactive tooltips**: Hover or click any token to see:
  - Token probability and rank
  - Top 2 alternative tokens with their probabilities
  - Full vocabulary search with instant filtering
- **Token injection**: Click any alternative or search result to inject it and regenerate from that point
- **Temperature-scaled probabilities**: Accurate probability calculations with configurable temperature
- **Highlight modes**: Toggle between probability-based and rank-based coloring (rank capped at 5)

### 📝 **Text View**
- **Plain text display**: View generated content without token highlighting or visual indicators
- **Preserved whitespace**: Shows actual spaces, newlines, and tabs in the generated text
- **Special token visibility**: Newlines (↵), tabs (→), and space sequences (␣) remain visible
- **Consistent interaction**: Hover and click tokens to inspect details just like in Token View
- **View persistence**: Your selected view (Token/Text/Diff) persists across page refreshes

### 🔀 **Diff View (DiffLens)**
- **Cross-model comparison**: Compare token probabilities between two different models on the same conversation
- **Probability diff highlighting**: Tokens colored by probability difference (green = generation model prefers, red = analysis model prefers)
- **Rank diff mode**: Alternative view showing which model ranks tokens higher
- **Cross-tokenizer support**: Handles models with different tokenizers via text-based alignment
- **Context-aware analysis**: Full conversation history (including system prompt) passed to both models

### 📊 **Logit Lens**
- **Layer-by-layer analysis**: Inspect what the model predicts at each transformer layer
- **Interactive heatmap**: Classic Logit Lens view with tokens on X-axis, layers on Y-axis
- **Probability evolution chart**: Track how specific token probabilities change across layers
  - Default: top 2 predictions from final layer
  - Search entire vocabulary to add any token to the chart
  - Interactive Chart.js visualization with hover details
- **Multi-token windows**: Analyze up to 20 consecutive tokens at once

### 💬 **Conversation Management**
- **Tree-structured conversations**: Full branching support with navigation between alternative responses
- **Edit and regenerate**: Edit any user message and regenerate responses with full history preserved
- **System prompts**: Set conversation-wide context with collapsible system prompt bubble
- **Persistent storage**: Conversations auto-save to browser localStorage
- **Multi-model support**: Switch between models mid-conversation

### 🎛️ **Generation Controls**
- **Sampling filters**: Top-K and Top-P (nucleus sampling) applied before token selection
  - Filter token distribution to control randomness and quality
  - Settings persist across sessions
- **Automation rules**: Define rules that trigger during generation
  - Criteria: probability threshold, consecutive low probability, text match
  - Actions: resample from same model, borrow from another model, inject static text
  - Track resampling chains with visual indicators
  - Show original and replacement tokens in annotations
- **System prompts**: Set conversation-wide context with persistent defaults
- **Force Assistant Start**: Prefill the beginning of the assistant's response (with visual indicators)
- **Temperature control**: Adjust sampling randomness (0 = greedy, higher = more random)
- **Streaming generation**: Real-time token-by-token streaming with SSE
- **Stop generation**: Cancel generation mid-stream with proper cleanup

### 🎨 **User Interface**
- **Monochromatic design**: Clean, minimal black-and-white aesthetic with grid background
- **Responsive layout**: Three-column layout (conversations sidebar, main chat, inspector sidebar)
- **Organized menu bar**: Logical grouping - File, View, Generation, Analysis
- **Smart tooltips**: Auto-positioning to stay within viewport bounds
- **Smooth animations**: Polished dropdown transitions and hover effects
- **Persistent preferences**: View modes, sampling settings, and rules saved across sessions

---

## Technical Architecture

### Backend (`app.py`)
- **Flask**: Web server with SSE streaming support
- **PyTorch + Transformers**: Local model inference with CUDA acceleration
- **Memory management**: Automatic GPU cache clearing, OOM error handling
- **Float16 optimization**: Efficient GPU memory usage with half-precision
- **Multiple endpoints**:
  - `/stream`: Token generation with streaming
  - `/api/logit-lens`: Layer-by-layer analysis
  - `/api/analyze-difflens`: Cross-model comparison
  - `/api/calculate-logprobs`: Probability calculation for existing text
  - `/api/search-tokens`: Full vocabulary search

### Frontend (`templates/chat.html`)
- **Vanilla JavaScript**: No framework dependencies
- **Chart.js**: Interactive probability evolution charts
- **Server-Sent Events**: Real-time streaming without WebSockets
- **LocalStorage**: Client-side conversation persistence
- **Tree data structure**: Efficient branching conversation management

### Styling (`static/chat.css`)
- **CSS custom properties**: Consistent theming with CSS variables
- **Flexbox & Grid**: Modern layout techniques
- **Smooth transitions**: Polished micro-interactions
- **Responsive design**: Adapts to different screen sizes

---

## Installation

### Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA support (recommended)
- ~16GB RAM minimum (more for larger models)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/stevenmolotnikov/Scope.git
cd Scope
```

2. **Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux
```

3. **Install dependencies**
```bash
pip install accelerate Flask transformers matplotlib

# Install PyTorch with CUDA support (for GPU acceleration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Use cu118 instead of cu121 if you have CUDA 11.8
```

4. **Run the application**
```bash
python app.py
```

5. **Open in browser**
```
http://localhost:5000
```

The first time you use a model, it will download from HuggingFace (~several GB per model). Subsequent loads use the cached version.

---

## Supported Models

Currently configured models (easily extensible):
- **Llama 3.1 8B Instruct** (`meta-llama/Llama-3.1-8B-Instruct`)
- **Gemma 2 2B IT** (`google/gemma-2-2b-it`)
- **Gemma 3 1B IT** (`google/gemma-3-1b-it`)

To add more models, update the `<select>` elements in `templates/chat.html`.

---

## Usage Guide

### Basic Chat
1. Select a model from the dropdown
2. Set temperature (0 for deterministic, 1.0 for standard sampling)
3. Optionally configure sampling filters (Generation → Sampling)
   - Top-K: Keep only top K most probable tokens
   - Top-P: Cumulative probability threshold (nucleus sampling)
4. Type your message and press Enter or click Send
5. Hover over any token to see its probability and alternatives

### View Modes
- **Token View**: Color-coded tokens by probability or rank
- **Text View**: Plain text display with special characters visible
- **Diff View**: Compare token probabilities between two models
- Switch between views using the View menu - your selection persists across refreshes

### Token Injection
1. Click any token to open the interactive tooltip
2. Search for alternative tokens in the search box
3. Click any alternative to inject it and regenerate from that point
4. The conversation branches, preserving the original response

### Logit Lens Analysis
1. Click any token to open the tooltip
2. Click "Open Logit Lens" button
3. Toggle between heatmap (layer predictions per token) and chart (probability evolution)
4. In chart mode, search and add tokens to track them across layers

### Model Comparison (DiffLens)
1. Generate a conversation with one model (the "generation model")
2. Enable DiffLens in Analysis menu
3. Select a different analysis model
4. Click "Apply Diff"
5. Tokens are now colored by probability difference between models
6. Toggle highlight mode in View menu to see probability diff vs. rank diff

### Automation Rules
1. Open Generation → Automation Rules
2. Add rules that trigger during generation:
   - **Criteria**: Probability below threshold, consecutive low probability, or text match
   - **Actions**: Resample from same model, borrow from other model, or inject text
3. Each rule shows enabled count badge in menu
4. Rules apply in real-time during generation
5. Tokens modified by rules show gear icon (⚙) and annotation with resampling chain

### Sampling Configuration
1. Open Generation → Sampling
2. Configure filters applied before token selection:
   - **Top-K**: Limit to top K most probable tokens (0 = disabled)
   - **Top-P**: Cumulative probability threshold for nucleus sampling (1.0 = disabled)
3. Settings save automatically and persist across sessions

### Force Assistant Start
1. Enable prefilling in Analysis menu
2. Enter text in the prefill input box
3. This text will prefill the assistant's response
4. Prefilled tokens are shown with dashed borders
5. Useful for testing refusal bypasses, steering responses, or testing continuations

### System Prompts
1. Open Generation → System Prompt
2. Enter your system prompt text
3. Click "Save as Default" to persist across sessions
4. The system prompt appears as a collapsible bubble at the top of the conversation
5. It's prepended to all model calls for context-setting

---

## Performance Tips

### GPU Memory Management
- **Use smaller models first**: Gemma-2-2B and Gemma-3-1B are great for testing
- **Watch for OOM errors**: The app will auto-clear cache and show friendly error messages
- **Start fresh conversations**: Long context uses more memory
- **Monitor GPU usage**: Check console output for memory allocation logs

### Model Loading
- **First load is slow**: Models download from HuggingFace (one-time)
- **Subsequent loads are fast**: Models are cached locally
- **GPU detection**: Check console for "CUDA is available!" on startup

---

## Development Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Core chat UI, streaming tokens, tooltips | ✅ |
| 2 | Logit Lens, DiffLens, branching conversations | ✅ |
| 3 | System prompts, highlight modes, memory optimization | ✅ |
| 4 | Sampling filters, automation rules, text view | ✅ |
| 5 | Export conversations, annotations, advanced filters | Planned |
| 6 | Hallucination detection, calibration analysis | Planned |
| 7 | TransformerLens integration, steering vectors | Planned |

---

## Contributing

Contributions are welcome! Areas of interest:
- **New analysis modes**: Entropy tracking, attention visualization, activation patching
- **Model integrations**: API providers (OpenAI, Anthropic), more local models
- **Export features**: CSV/JSON export, annotation tools, sharing
- **Performance**: Batching, caching, streaming optimizations
- **UI improvements**: Dark mode, accessibility, mobile support

**To contribute:**
1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Test with multiple models and conversation types
5. Submit a pull request with description and screenshots

---

## Project Structure

```
Scope/
├── app.py                    # Flask backend, model inference, streaming, rules
├── requirements.txt          # Python dependencies
├── templates/
│   ├── chat.html            # Main chat interface (5200+ lines)
│   └── index.html           # Legacy analysis interface
├── static/
│   ├── chat.css             # Main stylesheet (2800+ lines)
│   └── style.css            # Legacy styles
├── probability_monitor.py   # CLI probability inspection tool
└── README.md                # This file
```

---

## Troubleshooting

### CUDA Out of Memory
- Use a smaller model (Gemma-2-2B or Gemma-3-1B)
- Start a new conversation to reduce context length
- Reduce the Logit Lens analysis window size
- Close other GPU-intensive applications

### Model Not Loading
- Check internet connection (first download)
- Ensure sufficient disk space (~8GB per model)
- Verify HuggingFace is accessible
- Check console for detailed error messages

### Slow Generation
- First generation is slower (model loading)
- Ensure CUDA is properly installed and detected
- Check console for "Using device: cuda" message
- Consider using a smaller model for faster inference

### Conversation Not Saving
- Check browser console for localStorage errors
- Ensure localStorage is not disabled
- Try a different browser
- Check that you're not in private/incognito mode

---

## Citation

If you use Scope in your research, please cite:

```bibtex
@software{scope2024,
  title = {Scope: A Research Interface for Language Model Interpretability},
  author = {Molotnikov, Steven},
  year = {2024},
  url = {https://github.com/stevenmolotnikov/Scope}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/transformers/) - Model library
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Chart.js](https://www.chartjs.org/) - Charting library

Inspired by research in mechanistic interpretability, AI safety, and the need for better tools to understand language model behavior at the token level.

---

**Scope aims to become the standard interface for empirical alignment work. If you have a workflow that relies on token-level signals, reach out—we'd love to build it in.**
