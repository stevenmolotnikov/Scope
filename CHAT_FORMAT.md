# Chat Format Implementation

## Overview

TokenLens now properly uses each model's native chat template format for multi-turn conversations.

## How It Works

### 1. Message History
The frontend maintains a conversation history array:
```javascript
conversationHistory = [
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi there!"},
  {"role": "user", "content": "How are you?"}
]
```

### 2. Chat Template Application
When generating a response, the backend:
1. Receives the full conversation history
2. Applies the model's chat template via `tokenizer.apply_chat_template()`
3. Adds the generation prompt automatically

### 3. Model-Specific Formatting

#### Llama 3.1 Format
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|><|start_header_id|>user<|end_header_id|>

How are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

#### Gemma Format
```
<start_of_turn>user
Hello!<end_of_turn>
<start_of_turn>model
Hi there!<end_of_turn>
<start_of_turn>user
How are you?<end_of_turn>
<start_of_turn>model

```

## Key Features

### Automatic Template Detection
- Uses `tokenizer.chat_template` if available
- Falls back to raw text if no template exists
- Prints template info on model load for debugging

### Proper EOS Handling
The generator checks for multiple EOS tokens:
- Standard `tokenizer.eos_token_id`
- Model-specific tokens like `<|eot_id|>` for Llama
- `<|end_of_text|>` and `</s>` variants

### Token Probability Tracking
Even with chat templates, we track probabilities for:
- The actual generated tokens (not the template tokens)
- All special tokens are properly decoded
- Alternatives consider the full vocabulary

## Implementation Details

### Backend (`app.py`)
```python
def generate_streaming_tokens(model, tokenizer, device, messages, ...):
    # Apply chat template
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        # Fallback for models without templates
        formatted_prompt = messages[-1]['content']
```

### Frontend (`chat.html`)
```javascript
// Maintain conversation history
conversationHistory.push({role: 'user', content: message});

// Send full history to backend
fetch('/stream', {
    method: 'POST',
    body: JSON.stringify({
        messages: conversationHistory,
        model: modelName
    })
});

// Track assistant response
conversationHistory.push({role: 'assistant', content: assistantResponse});
```

## Testing

To verify proper chat formatting:

1. **Check console output** - The backend prints:
   - Tokenizer class name
   - Whether chat template exists
   - First 100 chars of template
   - Formatted prompt preview

2. **Multi-turn test**:
   ```
   User: "Remember that my name is Alice."
   Assistant: [responds]
   User: "What's my name?"
   Assistant: [should remember "Alice"]
   ```

3. **Context test**:
   ```
   User: "Let's count: 1, 2, 3"
   Assistant: [responds]
   User: "What comes next?"
   Assistant: [should say "4"]
   ```

## Supported Models

### With Chat Templates ✅
- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-2-2b-it`
- `google/gemma-3-1b-it`
- Most `-Instruct` or `-it` variants

### Without Chat Templates ⚠️
- Base models (non-instruct)
- Falls back to simple text continuation

## Debugging

If chat format isn't working:

1. Check server console for template info
2. Look for "Using chat template" vs "No chat template" messages
3. Verify message format in POST request (browser dev tools)
4. Check EOS token detection in generation logs

## Future Enhancements

- [ ] System message support
- [ ] Custom chat templates
- [ ] Template override options
- [ ] Multi-model comparison with different templates

