# TokenLens Changelog

## [Latest] - Proper Chat Format Implementation

### Fixed: Chat Template Support 🎯

**Problem**: The initial implementation was sending raw user messages without proper formatting for instruction-tuned models. Models like Llama-3.1-8B-Instruct expect specific chat templates with role markers.

**Solution**: Implemented proper chat template handling with conversation history tracking.

### Changes Made

#### Backend (`app.py`)

1. **Chat Template Application**
   - `generate_streaming_tokens()` now accepts a list of message dictionaries
   - Automatically applies model-specific chat templates via `tokenizer.apply_chat_template()`
   - Falls back to raw text for models without templates
   - Prints template information for debugging

2. **Enhanced Tokenizer Loading**
   - Sets `pad_token` if missing (defaults to `eos_token`)
   - Logs tokenizer class, chat template availability
   - Shows template preview on load

3. **Improved EOS Detection**
   - Checks for standard `eos_token_id`
   - Adds model-specific EOS tokens (`<|eot_id|>`, `<|end_of_text|>`, `</s>`)
   - Logs EOS detection for debugging

4. **Updated Stream Endpoint**
   - Changed from GET to POST for proper JSON payload
   - Accepts `messages` array and `model` parameter
   - Better error handling with traceback logging

#### Frontend (`chat.html`)

1. **Conversation History Tracking**
   - Maintains `conversationHistory` array with role-based messages
   - Automatically appends user and assistant messages
   - Preserves context across multiple turns

2. **Updated Request Method**
   - Changed from EventSource (GET) to fetch with streaming (POST)
   - Sends full conversation history with each request
   - Properly handles Server-Sent Events via ReadableStream

3. **Clear Chat Function**
   - Resets conversation history along with UI
   - Ensures fresh context for new conversations

### Model Support

#### ✅ Full Support (With Chat Templates)
- `meta-llama/Llama-3.1-8B-Instruct`
  - Format: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>...<|eot_id|>`
- `google/gemma-2-2b-it`
  - Format: `<start_of_turn>user....<end_of_turn><start_of_turn>model....`
- `google/gemma-3-1b-it`
  - Format: Similar to Gemma-2

#### ⚠️ Basic Support (No Chat Template)
- Base models fall back to simple text continuation
- Still functional but without multi-turn context awareness

### Testing

**Multi-turn conversation test:**
```
User: "My favorite color is blue."
AI: [responds]
User: "What's my favorite color?"
AI: [should remember "blue"]
```

**Console output example:**
```
Loading model: meta-llama/Llama-3.1-8B-Instruct
Tokenizer: LlamaTokenizerFast
Has chat template: True
Chat template preview: {% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start...
Using chat template. Formatted prompt: <|begin_of_text|><|start_header_id|>user<|end_header_id|>

My favorite color is blue.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

...
```

### Documentation Added

- `CHAT_FORMAT.md` - Detailed explanation of chat template implementation
- Updated `README.md` - Reflects chat interface as primary feature
- Updated `GETTING_STARTED.md` - Includes usage examples

### Breaking Changes

⚠️ The `/stream` endpoint changed from GET to POST. If you have custom integrations, update them to:
```javascript
fetch('/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        messages: [{role: 'user', content: 'Hello'}],
        model: 'meta-llama/Llama-3.1-8B-Instruct'
    })
})
```

### What This Enables

✅ **Multi-turn conversations**: Model remembers previous messages  
✅ **Proper instruction following**: Templates provide clear role separation  
✅ **Model-native formatting**: Each model uses its trained format  
✅ **Context preservation**: Full history sent with each request  
✅ **Better responses**: Instruction-tuned models work as intended  

### Next Steps

Based on `light_spec.md`, upcoming features:
- [ ] Token interference (force specific tokens from top-3)
- [ ] Model diffing (compare outputs side-by-side)
- [ ] Prefilling support (manipulate assistant start)
- [ ] Interpretability hooks (NDIF integration)

---

## Previous Versions

### [Initial Release] - ChatGPT-Style Interface

- Created streaming chat interface
- Real-time token probability visualization
- Color-coded tokens (red = low prob, green = high prob)
- Hover tooltips with detailed token information
- Model selection dropdown
- Legacy interface preserved at `/old` route

