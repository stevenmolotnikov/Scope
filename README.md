# TokenLens

A ChatGPT-style interface with real-time token probability visualization for AI research and model analysis.

## Motivation

In AI control settings, an untrusted but capable model (UTM) receives both a benign primary task and a hidden malicious side task, while a trusted but weaker model (TM) monitors for misbehavior. Existing control approaches rely on prompting the TM to detect suspicious outputs, but this requires the TM to compress and interpret increasing information as tasks grow complex, resulting in poor performance even on overt misbehavior in environments like ControlArena and SHADE-Arena.

**Can we shift from semantic interpretation of increasingly large texts to token-level control?**

This tool introduces an alternative control paradigm grounded in the observation that models form increasingly similar representations of the world as they become stronger (the Platonic Representation Hypothesis). When a UTM behaves honestly, its generated tokens should be plausible under the TM. However, side-task tokens may be out-of-distribution for the TM, since it never received the malicious subtask. 

Rather than asking *"Is there something suspicious in this output?"*, we ask *"What is the probability that a TM given the main task would generate this token?"*

## Features

- **ChatGPT-style interface**: Interactive chat with streaming token generation
- **Real-time token probabilities**: See probability distributions as tokens are generated
- **Color-coded visualization**: Tokens colored from red (low probability) to green (high probability)
- **Hover tooltips**: Detailed probability, rank, and top alternatives for each token
- **Model selection**: Choose from multiple models (Llama, Gemma, etc.)
- **Legacy mode**: Access original analysis tools at `/old` route

## Installation

```bash
pip install torch transformers flask
```

## Usage

### Command Line Analysis

Generate token probability data from prompt and output files:

```bash
python probability_monitor.py prompts/my_prompt.txt outputs/my_output.txt --model_name meta-llama/Llama-3.1-8B-Instruct
```

This saves a JSON file to the `probabilities/` directory containing per-token probability and rank data.

### Web Interface (ChatGPT-Style)

Launch the chat interface:

```bash
python app.py
```

Navigate to `http://localhost:5001` to:
- **Chat with models**: Send messages and see streaming responses with token probabilities
- **Inspect tokens**: Hover over any token to see:
  - Exact probability value
  - Rank in the vocabulary
  - Top 3 alternative tokens the model considered
- **Model selection**: Switch between different models in the header
- **Conversation history**: All chats saved to `conversations/` directory
- **Shareable links**: Each conversation has its own URL
- **Clear chat**: Delete conversations anytime

### Legacy Interface

The original analysis interface is still available at `http://localhost:5001/old`:
- **Load File**: View existing probability analyses
- **New Analysis**: Run direct analysis on prompt/output pairs
- **Adjust thresholds**: Configure probability/rank thresholds for flagging tokens

### Conversation Storage

Conversations are automatically saved to the `conversations/` directory as JSON files:

```
conversations/
├── conv_1699999999.json
├── conv_1700000000.json
└── conv_1700000001.json
```

Each file contains:
- Full message history (user and assistant)
- Token probability data for each response
- Model name and timestamp
- Conversation title

**Benefits:**
- Survives browser cache clearing
- Accessible from any browser/device
- Easy to backup, share, or analyze
- Each conversation has a unique URL: `http://localhost:5001/chat/conv_1699999999`

### Output Format

The tool generates JSON files with the following structure:

```json
{
  "prompt": "The original prompt text",
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "vocab_size": 128256,
  "tokens": [
    {
      "token": "Hello",
      "probability": 0.856234,
      "rank": 1,
      "top_alternatives": [...]
    }
  ]
}
```

## How It Works

1. **Tokenization**: The prompt and output are tokenized separately
2. **Forward Pass**: The model computes logits for the concatenated sequence
3. **Probability Calculation**: For each output token, we compute:
   - The probability assigned to the actual token
   - The rank of that token (1 = most likely)
   - Top alternative tokens the model preferred
4. **Visualization**: Tokens are color-coded from red (low probability) to green (high probability)

## Control Strategy

This tool enables a token-level control strategy:

1. Run the UTM's output through the TM to get per-token probabilities
2. Flag tokens with unusually low probability under the TM
3. Investigate flagged regions for potential malicious behavior

Since the TM only sees the benign task, tokens related to the hidden malicious task should be out-of-distribution and receive low probabilities.