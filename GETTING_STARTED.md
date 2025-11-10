# Getting Started with TokenLens

## Quick Start

TokenLens is now a ChatGPT-style interface with real-time token probability visualization!

### 1. Start the Server

```bash
python app.py
```

The server will start on `http://localhost:5001`

### 2. Open Your Browser

Navigate to `http://localhost:5001` - you'll see the chat interface with:
- **TokenLens** logo and model selector at the top
- A welcome message in the center
- An input box at the bottom

### 3. Chat with Token Probabilities

1. **Type your message** in the input box at the bottom
2. **Press Enter or click the send button**
3. **Watch the response stream** - tokens appear one by one with color coding:
   - 🟢 **Green**: High probability tokens (model is confident)
   - 🟡 **Yellow/Orange**: Medium probability tokens
   - 🔴 **Red**: Low probability tokens (model is uncertain)

### 4. Explore Token Details

**Hover over any token** to see:
- The exact token string
- Probability value (0.0 to 1.0)
- Rank in the vocabulary (1 = most likely)
- Top 3 alternative tokens the model considered

### 5. Switch Models

Click the **model dropdown** in the header to switch between:
- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-2-2b-it`
- `google/gemma-3-1b-it`

### 6. Clear Chat

Click the **Clear Chat** button to start a new conversation.

## Example Use Cases

### 1. Understanding Model Confidence
Ask: "What is the capital of France?"
- High-confidence tokens will be bright green
- Factual responses typically have high probabilities

### 2. Exploring Uncertainty
Ask: "What will happen in 2050?"
- Speculative content shows more varied colors
- Lower probabilities indicate uncertainty

### 3. Comparing Responses
- Clear chat
- Ask the same question to different models
- Compare token probabilities between models

## Features

### Real-Time Streaming
Tokens appear as they're generated - you don't wait for the full response.

### Color-Coded Probabilities
The background color of each token indicates its probability:
- HSL color scale from red (0°) to green (120°)
- Brightness and saturation optimized for readability

### Detailed Tooltips
Every token has a rich tooltip showing:
- Raw token string (including special characters)
- Exact probability (6 decimal places)
- Rank out of entire vocabulary
- Top 3 alternatives with their probabilities

### Clean Token Display
Special characters are visualized:
- `·` (middle dot) for spaces
- `↵` (return symbol) for newlines
- Normal display for all other text

## Legacy Interface

The original analysis interface is still available at `http://localhost:5001/old`:
- Load pre-computed probability files
- Analyze static prompt/output pairs
- Adjust probability/rank thresholds
- View flagged tokens in a table

## Technical Notes

### Models
The first time you use a model, it will be downloaded from HuggingFace. This may take several minutes depending on your internet connection.

Models are cached, so subsequent uses are instant.

### Hardware Requirements
- **GPU recommended**: Significantly faster generation
- **CPU mode**: Works but slower (especially for larger models)
- **Memory**: At least 8GB RAM for 8B models

### Generation Settings
Currently using greedy decoding (always picking the highest probability token). This ensures:
- Deterministic outputs
- Accurate probability tracking
- Faster generation

Future versions will support temperature-based sampling while still showing probabilities.

## Troubleshooting

### "Connection error" message
- Check that the Flask app is running
- Refresh the browser page
- Check console for errors

### Slow generation
- First model load takes time (downloading weights)
- CPU generation is slower than GPU
- Consider using smaller models (gemma-3-1b-it)

### Model not found
- Ensure you have internet connection for first download
- Check HuggingFace model name is correct
- Verify you have sufficient disk space

## Next Steps

Based on the Perplex spec, upcoming features include:
- [ ] Token injection (force specific tokens)
- [ ] Model comparison mode (side-by-side)
- [ ] Prefilling support
- [ ] Custom token sets (R-Set/C-Set for safety research)
- [ ] Session saving and annotations
- [ ] Batch processing

Enjoy exploring token probabilities in real-time! 🎉


