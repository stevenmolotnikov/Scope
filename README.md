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
- **Force Assistant Start**: Prefill the beginning of the assistant's response (with visual indicators)
- **Temperature control**: Adjust sampling randomness (0 = greedy, higher = more random)
- **Streaming generation**: Real-time token-by-token streaming with SSE
- **Stop generation**: Cancel generation mid-stream with proper cleanup

### 🎨 **User Interface**
- **Monochromatic design**: Clean, minimal black-and-white aesthetic with grid background
- **Responsive layout**: Three-column layout (conversations sidebar, main chat, tools sidebar)
- **Collapsible sections**: Organized tool categories (Conversation Settings, Views, Analysis Tools, Generation Tools)
- **Smart tooltips**: Auto-positioning to stay within viewport bounds
- **Smooth animations**: Polished dropdown transitions and hover effects

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
3. Type your message and press Enter or click Send
4. Hover over any token to see its probability and alternatives

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
2. In the sidebar under "Analysis Tools", select a different model
3. Click "Apply Analysis"
4. Tokens are now colored by probability difference between models
5. Toggle highlight mode to see probability diff vs. rank diff

### Force Assistant Start
1. In the sidebar under "Generation Tools", enter text in "Force Assistant Start"
2. This text will prefill the assistant's response
3. Prefilled tokens are shown with dashed borders
4. Useful for testing refusal bypasses, steering responses, or testing continuations

### System Prompts
1. In the sidebar under "Conversation Settings", enter a system prompt
2. Click "Save as Default" to persist across sessions
3. The system prompt appears as a collapsible bubble at the top of the conversation
4. It's prepended to all API calls for context-setting

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
| 4 | Export conversations, annotations, advanced filters | Planned |
| 5 | Hallucination detection, calibration analysis | Planned |
| 6 | TransformerLens integration, steering vectors | Planned |

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
├── app.py                    # Flask backend, model inference, streaming
├── requirements.txt          # Python dependencies
├── templates/
│   ├── chat.html            # Main chat interface (3300+ lines)
│   └── index.html           # Legacy analysis interface
├── static/
│   ├── chat.css             # Main stylesheet (1800+ lines)
│   └── style.css            # Legacy styles
├── probability_monitor.py   # CLI probability inspection tool
└── README.md               # This file
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
