from flask import Flask, render_template, url_for, request, redirect, flash, session, Response, stream_with_context, jsonify
import json
import os
import math
import datetime
import re
import secrets
import time
from collections import deque

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration
OUTPUT_DATA_DIR = "probabilities"
DEFAULT_PROB_THRESHOLD = 1e-4
DEFAULT_RANK_THRESHOLD = 100
DEFAULT_MAX_RANK_SCALING = 100
DEFAULT_MIN_PROB_SCALING = 0.0
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Global caches to avoid reloading
MODEL_CACHE = {}
TOKENIZER_CACHE = {}

SUPPORTED_RULE_CRITERIA = {
    'probability_below',
    'consecutive_probability_below',
    'text_match'
}

SUPPORTED_RULE_ACTIONS = {
    'resample_same',
    'resample_other_model',
    'replace_text'
}

MAX_RULE_ITERATIONS = 4


def _load_tokenizer_instance(model_name, log_details=False):
    """Create a tokenizer instance with consistent settings."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        try:
            # Some transformer builds expose AutoTokenizer from a nested module
            from transformers.models.auto.tokenization_auto import AutoTokenizer
        except Exception as e:
            raise ImportError(
                "Failed to import AutoTokenizer from transformers. "
                "Please ensure the 'transformers' package is installed and up to date."
            ) from e
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure pad token exists for models that omit it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if log_details:
            print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    if log_details:
        print(f"Tokenizer: {tokenizer.__class__.__name__}")
        print(f"Has chat template: {tokenizer.chat_template is not None}")
        if tokenizer.chat_template:
            preview = tokenizer.chat_template[:120]
            print(f"Chat template preview: {preview}...")
        print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        if hasattr(tokenizer, 'eos_token_ids'):
            print(f"Multiple EOS token IDs: {tokenizer.eos_token_ids}")
        if getattr(tokenizer, 'additional_special_tokens', None):
            print(f"Additional special tokens (first 5): {tokenizer.additional_special_tokens[:5]}")
    
    return tokenizer


def get_tokenizer(model_name, log_details=False):
    """Retrieve a tokenizer from cache or load it if needed."""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name][1]
    
    if model_name in TOKENIZER_CACHE:
        return TOKENIZER_CACHE[model_name]
    
    tokenizer = _load_tokenizer_instance(model_name, log_details=log_details)
    TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer

def get_color_for_prob(probability, min_prob_for_scaling=0.0):
    """Maps probability (0-1) to an HSL color string.
    Low prob -> Red (0°), High prob -> Green (120°)
    Optionally compress the scale by setting min_prob_for_scaling > 0.
    """
    if probability is None or not (0 <= probability <= 1):
        return "hsl(0, 0%, 80%)"  # Grey for invalid
    
    # Optional: compress scale if min_prob_for_scaling is set
    if min_prob_for_scaling > 0:
        probability = max(probability, min_prob_for_scaling)
        # Rescale from [min, 1] to [0, 1]
        probability = (probability - min_prob_for_scaling) / (1 - min_prob_for_scaling)
    
    # Simple linear mapping: 0 -> red (0°), 1 -> green (120°)
    hue = 120 * probability
    
    return f"hsl({hue:.1f}, 70%, 60%)"


def get_color_for_rank(rank, vocab_size, max_rank_for_scaling=100):
    """Maps rank (1-vocab_size) to an HSL color string.
    Low rank (good) -> Green (120°), High rank (bad) -> Red (0°)
    Clamps at max_rank_for_scaling to avoid everything being red.
    """
    if rank is None or vocab_size is None or rank < 1:
        return "hsl(0, 0%, 80%)"  # Grey for invalid
    
    # Clamp rank to max_rank_for_scaling
    rank = min(rank, max_rank_for_scaling)
    
    # Convert to 0-1 scale: rank 1 -> 1.0 (best), rank max -> 0.0 (worst)
    normalized = 1.0 - ((rank - 1) / (max_rank_for_scaling - 1))
    
    # Map to hue: 0 -> red (0°), 1 -> green (120°)
    hue = 120 * normalized
    
    return f"hsl({hue:.1f}, 70%, 60%)"


def scan_available_files():
    """Scan for available JSON files in the output directory."""
    try:
        if not os.path.exists(OUTPUT_DATA_DIR):
            print(f"Creating data directory: {OUTPUT_DATA_DIR}")
            os.makedirs(OUTPUT_DATA_DIR)
        available_files = [f for f in os.listdir(OUTPUT_DATA_DIR) if f.endswith('.json')]
        available_files.sort()
        return available_files, None
    except Exception as e:
        return [], f"Error scanning directory {OUTPUT_DATA_DIR}: {e}"


def determine_file_to_load(available_files, selected_file):
    """Determine which file to load based on available files and selection."""
    if selected_file and selected_file in available_files:
        return os.path.join(OUTPUT_DATA_DIR, selected_file), selected_file
    elif not selected_file and available_files:
        # Load first file if available and none selected
        first_file = available_files[0]
        return os.path.join(OUTPUT_DATA_DIR, first_file), first_file
    return None, selected_file


def load_token_data(file_path):
    """Load and parse token data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return (data.get("prompt", "Prompt not found in JSON data."), 
                data.get("tokens", []), 
                data.get("vocab_size"), 
                data.get("model_name"),
                None)
    except FileNotFoundError:
        return None, [], None, None, f"File not found at {file_path}"
    except json.JSONDecodeError:
        return None, [], None, None, f"Error decoding {file_path}. Is it valid JSON?"
    except Exception as e:
        return None, [], None, None, f"An unexpected error occurred while processing {file_path}: {e}"


def clean_tokenizer_artifacts(token):
    """Clean up tokenizer artifacts from existing JSON data."""
    if not isinstance(token, str):
        token = str(token)
    
    # Remove common special tokens
    special_tokens_to_remove = [
        '<|begin_of_text|>', '<|end_of_text|>', 
        '<|start_header_id|>', '<|end_header_id|>',
        '<s>', '</s>', '<pad>', '<unk>'
    ]
    
    for special_token in special_tokens_to_remove:
        token = token.replace(special_token, '')
    
    # Clean up GPT-style tokenizer artifacts
    token = token.replace('Ġ', ' ')  # GPT space marker
    token = token.replace('Ċ', '\n')  # GPT newline marker
    token = token.replace('ĉ', '\t')  # GPT tab marker
    
    # Clean up other common artifacts
    token = token.replace('▁', ' ')  # SentencePiece space marker
    token = token.replace('<0x0A>', '\n')  # Hex newline
    token = token.replace('<0x20>', ' ')  # Hex space
    
    return token


def format_token_for_html(token):
    """Clean tokenizer artifacts and prepare for HTML display while making whitespace visible."""
    if not isinstance(token, str):
        token = str(token)
    
    # First clean tokenizer artifacts
    clean_token = clean_tokenizer_artifacts(token)
    
    # HTML escape the clean token
    escaped_token = clean_token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Make whitespace tokens visible while preserving their function
    if clean_token == ' ':
        return '·'  # Middle dot for space - visible but subtle
    elif clean_token == '\n':
        return '↵'  # Return symbol for newline
    elif clean_token == '\t':
        return '→'  # Right arrow for tab
    elif clean_token.isspace() and len(clean_token) > 1:
        # Multiple spaces or mixed whitespace - just show the visual representation
        result = ''
        for char in clean_token:
            if char == ' ':
                result += '·'
            elif char == '\n':
                result += '↵'
            elif char == '\t':
                result += '→'
            else:
                result += char  # Any other whitespace
        return result
    
    return escaped_token


def process_token_data(token_input_list, vocab_size, threshold_mode, max_rank_for_scaling, min_prob_for_scaling):
    """Process raw token data into formatted data for display."""
    processed_token_data = []
    
    for idx, item in enumerate(token_input_list):
        prob = item.get('probability')
        token = item.get('token', '')
        rank = item.get('rank')
        top_alternatives = item.get('top_alternatives', [])
        
        # Simple token processing - just format for HTML display
        display_token = format_token_for_html(token)
        token_class = determine_token_class(token)
        
        # Process alternatives for HTML display
        processed_alternatives = [
            {
                'token': format_token_for_html(alt.get('token', '')),
                'probability': alt.get('probability'),
                'rank': alt.get('rank')
            }
            for alt in top_alternatives
        ]
        
        # Choose color based on current mode
        token_color = (get_color_for_rank(rank, vocab_size, max_rank_for_scaling) 
                      if threshold_mode == 'rank' 
                      else get_color_for_prob(prob, min_prob_for_scaling))
        
        processed_item = {
            'probability': prob,
            'rank': rank,
            'vocab_size': vocab_size,
            'token': token,
            'color': token_color,
            'display_token': display_token,
            'token_class': token_class,
            'original_index': idx,
            'output_token_index': idx,
            'top_alternatives': processed_alternatives
        }
        processed_token_data.append(processed_item)
    
    return processed_token_data


def determine_token_class(token):
    """Determine CSS class for special tokens after cleaning artifacts."""
    # Clean the token first to check its actual content
    clean_token = clean_tokenizer_artifacts(token)
    
    if '\n' in clean_token or '\r' in clean_token:
        return "newline-token"
    elif clean_token.isspace() and clean_token != '':
        return "space-token"
    return ""


def apply_threshold_filtering(processed_token_data, threshold_mode, prob_threshold, rank_threshold):
    """Apply threshold filtering to highlight tokens."""
    if threshold_mode == 'rank':
        lowest_probs = [token for token in processed_token_data 
                       if token.get('rank') is not None and token['rank'] >= rank_threshold]
        lowest_probs.sort(key=lambda x: x['rank'], reverse=True)
    else:
        lowest_probs = [token for token in processed_token_data 
                       if token['probability'] is not None and token['probability'] < prob_threshold]
        lowest_probs.sort(key=lambda x: x['probability'])
    
    # Mark tokens that meet threshold criteria
    lowest_prob_indices = {item['original_index'] for item in lowest_probs}
    for item in processed_token_data:
        item['is_lowest_prob'] = (item['original_index'] in lowest_prob_indices)
    
    return processed_token_data, lowest_probs


def sanitize_generation_rules(rules_input):
    """Validate and normalize rule definitions received from the client."""
    sanitized = []
    if not isinstance(rules_input, list):
        return sanitized
    
    for idx, raw_rule in enumerate(rules_input):
        if not isinstance(raw_rule, dict):
            continue
        
        if not raw_rule.get('enabled', True):
            continue
        
        criteria = raw_rule.get('criteria') or {}
        action = raw_rule.get('action') or {}
        criteria_type = criteria.get('type')
        action_type = action.get('type')
        
        if criteria_type not in SUPPORTED_RULE_CRITERIA or action_type not in SUPPORTED_RULE_ACTIONS:
            continue
        
        rule_id = str(raw_rule.get('id') or f"rule_{idx + 1}")
        rule_name = raw_rule.get('name') or f"Rule {idx + 1}"
        rule_state = {}
        
        if criteria_type == 'consecutive_probability_below':
            rule_state['consecutive_hits'] = 0
        if criteria_type == 'text_match':
            window = max(1, int(criteria.get('window', 3)))
            rule_state['recent_tokens'] = deque(maxlen=max(window, 6))
        
        sanitized.append({
            'id': rule_id,
            'name': rule_name,
            'criteria': criteria,
            'action': action,
            'enabled': True,
            '_state': rule_state
        })
    
    return sanitized


def rule_matches_candidate(rule, candidate):
    """Check whether the current candidate token triggers the provided rule."""
    criteria = rule.get('criteria', {})
    ctype = criteria.get('type')
    state = rule.get('_state', {})
    probability = candidate.get('probability')
    token_text = candidate.get('token', '')
    
    if ctype == 'probability_below':
        threshold = float(criteria.get('threshold', 0.05))
        return probability is not None and probability < threshold
    
    if ctype == 'consecutive_probability_below':
        threshold = float(criteria.get('threshold', 0.05))
        required_count = max(2, int(criteria.get('count', 2)))
        current_hits = state.get('consecutive_hits', 0)
        if probability is not None and probability < threshold:
            return (current_hits + 1) >= required_count
        return False
    
    if ctype == 'text_match':
        target_value = criteria.get('value', '')
        if not target_value:
            return False
        
        window = max(1, int(criteria.get('window', 3)))
        recent_tokens = state.get('recent_tokens')
        if recent_tokens is None:
            recent_tokens = deque(maxlen=max(window, 6))
            state['recent_tokens'] = recent_tokens
        
        preview = list(recent_tokens)[- (window - 1):] if window > 1 else []
        preview.append(token_text or '')
        combined_text = ''.join(preview[-window:])
        match_type = criteria.get('match_type', 'contains')
        
        if match_type == 'exact':
            return combined_text == target_value
        if match_type == 'regex':
            pattern = state.get('_compiled_regex')
            if pattern is None:
                try:
                    pattern = re.compile(target_value)
                except re.error:
                    pattern = None
                state['_compiled_regex'] = pattern
            return bool(pattern.search(combined_text)) if pattern else False
        # Default to substring contains
        return target_value in combined_text
    
    return False


def update_rule_state_with_token(rule, token_text, probability):
    """Update per-rule state after a token has been finalized."""
    criteria = rule.get('criteria', {})
    ctype = criteria.get('type')
    state = rule.get('_state', {})
    
    if ctype == 'consecutive_probability_below':
        threshold = float(criteria.get('threshold', 0.05))
        if probability is not None and probability < threshold:
            state['consecutive_hits'] = state.get('consecutive_hits', 0) + 1
        else:
            state['consecutive_hits'] = 0
    
    if ctype == 'text_match':
        recent_tokens = state.get('recent_tokens')
        if recent_tokens is not None:
            recent_tokens.append(token_text or '')


def select_from_distribution(probabilities_row, strategy='sample'):
    """Select a token ID from a probability vector using the requested strategy."""
    import torch
    
    if probabilities_row.sum() <= 0:
        return None
    
    if strategy == 'greedy':
        return int(torch.argmax(probabilities_row).item())
    
    return int(torch.multinomial(probabilities_row, num_samples=1).item())


def execute_rule_action(rule, candidate, probabilities_row, tokenizer, rule_context):
    """Execute the action associated with a rule and return replacement tokens."""
    action_type = rule.get('action', {}).get('type')
    
    if action_type == 'resample_same':
        return _rule_action_resample_same(rule, candidate, probabilities_row, tokenizer, rule_context)
    if action_type == 'replace_text':
        return _rule_action_replace_text(rule, tokenizer, candidate)
    if action_type == 'resample_other_model':
        return _rule_action_resample_other_model(rule, candidate, probabilities_row, tokenizer, rule_context)
    
    return None


def _rule_action_resample_same(rule, candidate, probabilities_row, tokenizer, rule_context):
    """Resample a token from the active model distribution."""
    import torch
    
    action_config = rule.get('action', {})
    strategy = action_config.get('strategy', 'sample')
    top_k = int(action_config.get('top_k', 0) or 0)
    max_attempts = max(1, int(action_config.get('max_attempts', 3)))
    
    filtered = probabilities_row.clone()
    if top_k > 0 and top_k < filtered.size(0):
        top_probs, top_indices = torch.topk(filtered, top_k)
        mask = torch.zeros_like(filtered)
        mask[top_indices] = filtered[top_indices]
        filtered = mask
    
    banned_ids = {candidate.get('token_id')}
    
    for attempt in range(max_attempts):
        sample_probs = filtered.clone()
        for banned in list(banned_ids):
            if 0 <= banned < sample_probs.size(0):
                sample_probs[banned] = 0
        
        total = float(sample_probs.sum())
        if total <= 0:
            break
        sample_probs = sample_probs / total
        
        sampled_id = select_from_distribution(sample_probs, strategy=strategy)
        if sampled_id is None:
            break
        
        banned_ids.add(sampled_id)
        if sampled_id == candidate.get('token_id'):
            continue
        
        new_token = tokenizer.decode([sampled_id])
        rank_tensor = rule_context.get('rank_tensor')
        new_rank = None
        if rank_tensor is not None:
            matches = (rank_tensor == sampled_id).nonzero(as_tuple=True)
            if matches and len(matches[0]) > 0:
                new_rank = int(matches[0][0].item() + 1)
        if new_rank is None:
            rank_lookup = rule_context.get('rank_lookup') or {}
            new_rank = rank_lookup.get(int(sampled_id))
        replacement = {
            'token_id': int(sampled_id),
            'token': new_token,
            'probability': float(probabilities_row[sampled_id].item()),
            'rank': new_rank,
            'top_alternatives': candidate.get('top_alternatives', [])
        }
        
        return {
            'replacement_tokens': [replacement],
            'reason': f"Resampled from base model (attempt {attempt + 1})"
        }
    
    return None


def _rule_action_replace_text(rule, tokenizer, candidate=None):
    """Replace the current token with static user-provided text."""
    action_config = rule.get('action', {})
    replacement_text = action_config.get('text', '')
    if not replacement_text:
        return None
    
    token_ids = tokenizer.encode(replacement_text, add_special_tokens=False)
    if not token_ids:
        return None
    
    replacement_tokens = []
    rank_lookup = rule_context.get('rank_lookup') or {}
    for idx, token_id in enumerate(token_ids):
        replacement_tokens.append({
            'token_id': int(token_id),
            'token': tokenizer.decode([int(token_id)]),
            'probability': None,
            'rank': None,
            'top_alternatives': candidate.get('top_alternatives', []) if candidate and idx == 0 else []
        })
    
    return {
        'replacement_tokens': replacement_tokens,
        'reason': 'Static replacement text injected'
    }


def _rule_action_resample_other_model(rule, candidate, probabilities_row, tokenizer, rule_context):
    """Use an alternate model to suggest the next token."""
    replacement_ids = _sample_from_alternate_model(rule, rule_context, tokenizer)
    if not replacement_ids:
        return None
    
    rank_lookup = rule_context.get('rank_lookup') or {}
    replacement_tokens = []
    
    rank_tensor = rule_context.get('rank_tensor')
    
    for idx, token_id in enumerate(replacement_ids):
        token_id_int = int(token_id)
        token_text = tokenizer.decode([token_id_int])
        rank_value = None
        if rank_tensor is not None:
            matches = (rank_tensor == token_id_int).nonzero(as_tuple=True)
            if matches and len(matches[0]) > 0:
                rank_value = int(matches[0][0].item() + 1)
        replacement_tokens.append({
            'token_id': token_id_int,
            'token': token_text,
            'probability': float(probabilities_row[token_id_int].item()) if token_id_int < probabilities_row.size(0) else None,
            'rank': rank_value or rank_lookup.get(token_id_int),
            'top_alternatives': candidate.get('top_alternatives', []) if idx == 0 else []
        })
    
    target_model = rule.get('action', {}).get('model') or 'alternate model'
    return {
        'replacement_tokens': replacement_tokens,
        'reason': f"Borrowed token from {target_model}"
    }


def _sample_from_alternate_model(rule, rule_context, tokenizer):
    """Sample a token from the specified alternate model and convert it into the base tokenizer space."""
    import torch
    
    action_config = rule.get('action', {})
    target_model_name = action_config.get('model')
    if not target_model_name:
        return []
    
    messages = rule_context.get('messages', [])
    generated_text = rule_context.get('generated_text', '')
    
    alt_model, alt_tokenizer, alt_device = load_model_and_tokenizer(target_model_name)
    
    if hasattr(alt_tokenizer, 'apply_chat_template') and alt_tokenizer.chat_template:
        formatted_prompt = alt_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        formatted_prompt = messages[-1]['content'] if messages else ''
    
    input_ids = alt_tokenizer.encode(formatted_prompt, return_tensors="pt").to(alt_device)
    
    if generated_text:
        assistant_ids = alt_tokenizer.encode(generated_text, return_tensors="pt", add_special_tokens=False).to(alt_device)
        if assistant_ids.size(1) > 0:
            input_ids = torch.cat([input_ids, assistant_ids], dim=1)
    
    with torch.no_grad():
        logits = alt_model(input_ids).logits[:, -1, :]
    
    strategy = action_config.get('strategy', 'greedy')
    temperature = float(action_config.get('temperature', 0.7))
    if temperature >= 0.01:
        logits = logits / temperature
    
    probabilities = torch.softmax(logits, dim=-1)[0]
    
    top_k = int(action_config.get('top_k', 0) or 0)
    if top_k > 0 and top_k < probabilities.size(0):
        top_probs, top_indices = torch.topk(probabilities, top_k)
        mask = torch.zeros_like(probabilities)
        mask[top_indices] = probabilities[top_indices]
        probabilities = mask
        total = probabilities.sum()
        if total > 0:
            probabilities = probabilities / total
    
    if probabilities.sum() <= 0:
        return []
    
    sampled_id = select_from_distribution(probabilities, strategy=strategy)
    if sampled_id is None:
        return []
    
    suggested_text = alt_tokenizer.decode([sampled_id])
    token_ids = tokenizer.encode(suggested_text, add_special_tokens=False)
    return token_ids


def apply_rules_to_candidate(candidate, rules, probabilities_row, tokenizer, rule_context):
    """Apply enabled rules to the current candidate token."""
    if not rules:
        return candidate, [], None
    
    pending_sequence = []
    applied_info = None
    iterations = 0
    
    while iterations < MAX_RULE_ITERATIONS:
        triggered_rule = None
        for rule in rules:
            if rule_matches_candidate(rule, candidate):
                triggered_rule = rule
                break
        
        if triggered_rule is None:
            break
        
        action_result = execute_rule_action(triggered_rule, candidate, probabilities_row, tokenizer, rule_context)
        if not action_result:
            break
        
        replacements = action_result.get('replacement_tokens') or []
        if not replacements:
            break
        
        candidate = replacements[0]
        
        if len(replacements) > 1:
            for extra in replacements[1:]:
                pending_sequence.append({
                    'token_id': extra['token_id'],
                    'token': extra.get('token')
                })
        
        applied_info = {
            'id': triggered_rule['id'],
            'name': triggered_rule['name'],
            'action': triggered_rule.get('action', {}).get('type'),
            'reason': action_result.get('reason')
        }
        
        if triggered_rule['criteria']['type'] == 'consecutive_probability_below':
            triggered_rule['_state']['consecutive_hits'] = 0
        
        iterations += 1
    
    return candidate, pending_sequence, applied_info


def load_model_and_tokenizer(model_name):
    """Load and cache model and tokenizer."""
    # Lazy import - only load these heavy libraries when needed
    import torch
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        try:
            from transformers.models.auto.modeling_auto import AutoModelForCausalLM
        except Exception as e:
            raise ImportError(
                "Failed to import AutoModelForCausalLM from transformers. "
                "Please ensure the 'transformers' package is installed and up to date."
            ) from e
    
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    
    print(f"Loading model: {model_name}")
    tokenizer = get_tokenizer(model_name, log_details=True)
    
    # Determine device with detailed logging
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"CUDA is available!")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = 'cpu'
        print(f"WARNING: CUDA not available, using CPU")
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch built with CUDA: {torch.version.cuda}")
    
    print(f"Loading model to device: {device}")
    
    # Clear CUDA cache before loading new model
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    # Enable memory-efficient features for CUDA
    if device == 'cuda':
        # Use gradient checkpointing to save memory (doesn't affect inference much)
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # Check model's generation config for EOS tokens
    if hasattr(model, 'generation_config') and hasattr(model.generation_config, 'eos_token_id'):
        gen_eos = model.generation_config.eos_token_id
        if isinstance(gen_eos, list):
            print(f"Model generation config EOS token IDs: {gen_eos}")
        else:
            print(f"Model generation config EOS token ID: {gen_eos}")
    
    # Model should already be on device via device_map, but ensure it
    if device == 'cpu':
        model.to(device)
    
    MODEL_CACHE[model_name] = (model, tokenizer, device)
    # Remove duplicate tokenizer entry now that it's bundled with the model
    TOKENIZER_CACHE.pop(model_name, None)
    return model, tokenizer, device


def calculate_token_probabilities(model, tokenizer, device, prompt, output, top_k=4):
    """Calculate token probabilities for the output text given the prompt."""
    # Lazy import - only load when needed
    import torch
    
    # Tokenize prompt and output separately
    prompt_encoding = tokenizer(prompt, return_tensors="pt")
    output_encoding = tokenizer(output, return_tensors="pt")
    
    prompt_ids = prompt_encoding["input_ids"][0].to(device)
    output_ids = output_encoding["input_ids"][0].to(device)
    
    # Concatenate for single forward pass
    all_ids = torch.cat([prompt_ids, output_ids]).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(all_ids).logits  # [1, total_len, vocab_size]
    
    prompt_len = prompt_ids.size(0)
    vocab_size = tokenizer.vocab_size
    normalized_rules = sanitize_generation_rules(rules or [])
    forced_token_queue = deque()
    generated_text_parts = []
    generated_text = ''
    token_data = []
    
    for i, token_id in enumerate(output_ids):
        token_id = token_id.item()  # Convert to regular int
        
        # Get logits for position that predicts current token
        current_logits = logits[0, prompt_len + i - 1]
        probabilities = torch.softmax(current_logits, dim=-1)
        
        # Get probability and rank of actual token
        actual_prob = probabilities[token_id].item()
        
        # Get raw token string (no cleaning - preserve original data)
        token_str = tokenizer.decode([token_id])
        
        # Calculate rank efficiently
        sorted_indices = torch.argsort(probabilities, descending=True)
        actual_rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1
        
        # Get top k alternatives (save raw tokens)
        top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
        top_k_tokens = [
            tokenizer.decode([idx.item()])
            for idx in top_k_indices
        ]
        
        top_alternatives = [
            {"token": token, "probability": prob.item(), "rank": idx + 1}
            for idx, (token, prob) in enumerate(zip(top_k_tokens, top_k_probs))
        ]
        
        token_data.append({
            "token": token_str,
            "probability": actual_prob,
            "rank": actual_rank,
            "top_alternatives": top_alternatives
        })
    
    return token_data, vocab_size


def generate_streaming_tokens(model, tokenizer, device, messages, max_new_tokens=512, top_k=4,
                              temperature=1.0, prefill=None, show_prompt_tokens=False,
                              rules=None, model_name=None):
    """Generate tokens one at a time with probabilities using streaming.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
                  e.g., [{"role": "user", "content": "Hello!"}]
        temperature: Sampling temperature (default 1.0)
        prefill: Optional text to prepend to assistant's response
        show_prompt_tokens: If True, yield prompt tokens (including special tokens) before generation
    """
    import torch
    
    # Apply chat template to format messages correctly for the model
    formatted_prompt = None
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        # Use the model's chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        print(f"Using chat template. Formatted prompt: {formatted_prompt[:200]}...")
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
    else:
        # Fallback: just use the last user message
        print("No chat template available, using raw message")
        prompt_text = messages[-1]['content'] if messages else ""
        formatted_prompt = prompt_text
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    
    # Store original prompt length before prefill
    original_prompt_length = input_ids.size(1)
    
    vocab_size = tokenizer.vocab_size
    
    # If prefill is provided, calculate probabilities efficiently with single forward pass
    if prefill:
        print(f"Prefilling with: {prefill[:100]}...")
        prefill_ids = tokenizer.encode(prefill, return_tensors="pt", add_special_tokens=False).to(device)
        
        # Single forward pass for all prefill tokens
        with torch.no_grad():
            all_ids = torch.cat([input_ids, prefill_ids], dim=1)
            logits = model(all_ids).logits  # [1, total_len, vocab_size]
        
        prompt_len = input_ids.size(1)
        
        # Calculate probabilities for each prefill token from single forward pass
        for i, token_id in enumerate(prefill_ids[0]):
            token_id_item = token_id.item()
            
            # Get logits for position that predicts this token
            current_logits = logits[0, prompt_len + i - 1]
            
            # Apply temperature scaling if not greedy
            if temperature != 0 and temperature >= 0.01:
                current_logits = current_logits / temperature
            
            probabilities = torch.softmax(current_logits, dim=-1)
            
            token_prob = probabilities[token_id_item].item()
            
            # Calculate rank
            sorted_indices = torch.argsort(probabilities, descending=True)
            token_rank = (sorted_indices == token_id_item).nonzero(as_tuple=True)[0].item() + 1
            
            # Get top k alternatives
            top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
            top_k_tokens = [tokenizer.decode([idx.item()]) for idx in top_k_indices]
            
            # Decode the actual token consistently
            decoded_token = tokenizer.decode([token_id_item])
            
            top_alternatives = [
                {"token": token, "probability": prob.item(), "rank": idx + 1}
                for idx, (token, prob) in enumerate(zip(top_k_tokens, top_k_probs))
            ]
            
            # Yield prefill token with probabilities
            yield {
                "token": decoded_token,
                "token_id": int(token_id_item),
                "probability": token_prob,
                "rank": token_rank,
                "vocab_size": vocab_size,
                "top_alternatives": top_alternatives,
                "is_prefill_token": True
            }
            generated_text_parts.append(decoded_token)
            generated_text += decoded_token
            if normalized_rules:
                for rule in normalized_rules:
                    update_rule_state_with_token(rule, decoded_token, token_prob)
        
        # Add prefill to input for generation
        input_ids = torch.cat([input_ids, prefill_ids], dim=1)
    
    # If requested, yield prompt tokens (to show special tokens like <start_of_turn>)
    if show_prompt_tokens:
        prompt_ids_to_show = input_ids[0][:original_prompt_length]
        for i, token_id in enumerate(prompt_ids_to_show):
            token_str = tokenizer.decode([token_id.item()])
            yield {
                'token': token_str,
                    'token_id': int(token_id.item()),
                'probability': None,
                'rank': None,
                'vocab_size': vocab_size,
                'top_alternatives': [],
                'is_prompt_token': True
            }
    
    # Get EOS token IDs directly from tokenizer configuration
    eos_token_ids = set()
    
    # Standard EOS token
    if tokenizer.eos_token_id is not None:
        eos_token_ids.add(tokenizer.eos_token_id)
    
    # Check if tokenizer has multiple EOS tokens (newer HuggingFace feature)
    if hasattr(tokenizer, 'eos_token_ids'):
        eos_ids = tokenizer.eos_token_ids
        if isinstance(eos_ids, (list, tuple)):
            eos_token_ids.update(eos_ids)
        elif eos_ids is not None:
            eos_token_ids.add(eos_ids)
    
    # Get generation config stop token IDs (most reliable for chat models)
    try:
        if hasattr(model, 'generation_config') and hasattr(model.generation_config, 'eos_token_id'):
            eos_from_config = model.generation_config.eos_token_id
            if isinstance(eos_from_config, (list, tuple)):
                eos_token_ids.update(eos_from_config)
            elif eos_from_config is not None:
                eos_token_ids.add(eos_from_config)
    except:
        pass
    
    print(f"EOS token IDs from tokenizer/model config: {sorted(eos_token_ids)}")
    if eos_token_ids:
        # Show what tokens these IDs correspond to
        eos_tokens_decoded = [tokenizer.decode([tid]) for tid in eos_token_ids]
        print(f"EOS tokens: {eos_tokens_decoded}")
    
    # Generate tokens one by one
    generated_ids = input_ids.clone()
    generated_text = ''.join(generated_text_parts)
    
    for step_idx in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]
        del outputs
        
        if temperature == 0 or temperature < 0.01:
            probabilities = torch.softmax(logits, dim=-1)
        else:
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)
        
        prob_row = probabilities[0]
        forced_entry = forced_token_queue.popleft() if forced_token_queue else None
        
        if forced_entry:
            next_token_id = int(forced_entry['token_id'])
        elif temperature == 0 or temperature < 0.01:
            next_token_id = torch.argmax(probabilities, dim=-1).item()
        else:
            next_token_id = torch.multinomial(probabilities, num_samples=1).item()
        
        if device == 'cuda' and step_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        token_prob = prob_row[next_token_id].item()
        if token_prob == 0.0 or token_prob < 1e-10:
            print(f"WARNING: Very low probability detected: {token_prob} for token '{tokenizer.decode([next_token_id])}'")
            print(f"  Logits shape: {logits.shape}, Probabilities shape: {probabilities.shape}")
            print(f"  Selected token ID: {next_token_id}, Temperature: {temperature}")
            print(f"  Top 5 probabilities: {prob_row.topk(5)}")
        
        sorted_indices = torch.argsort(prob_row, descending=True)
        rank_matches = (sorted_indices == next_token_id).nonzero(as_tuple=True)
        if rank_matches and len(rank_matches[0]) > 0:
            token_rank = int(rank_matches[0][0].item() + 1)
        else:
            token_rank = None
        
        top_k_probs, top_k_indices = torch.topk(prob_row, top_k)
        top_k_tokens = [tokenizer.decode([idx.item()]) for idx in top_k_indices]
        top_alternatives = [
            {"token": token, "probability": prob.item(), "rank": idx + 1}
            for idx, (token, prob) in enumerate(zip(top_k_tokens, top_k_probs))
        ]
        
        token_str = forced_entry.get('token') if forced_entry and forced_entry.get('token') is not None else tokenizer.decode([next_token_id])
        candidate_data = {
            "token_id": int(next_token_id),
            "token": token_str,
            "probability": float(token_prob),
            "rank": token_rank,
            "top_alternatives": top_alternatives
        }
        
        rule_applied_info = None
        rule_context = {
            'messages': messages,
            'generated_text': generated_text,
            'model_name': model_name,
            'temperature': temperature,
            'rank_tensor': sorted_indices,
            'rank_lookup': {}
        }
        
        if forced_entry:
            rule_applied_info = forced_entry.get('rule_info')
        elif normalized_rules:
            candidate_data, pending_sequence, rule_applied_info = apply_rules_to_candidate(
                candidate_data,
                normalized_rules,
                prob_row,
                tokenizer,
                rule_context
            )
            if pending_sequence:
                for extra in reversed(pending_sequence):
                    forced_id = int(extra['token_id'])
                    queued = {
                        'token_id': forced_id,
                        'token': extra.get('token') or tokenizer.decode([forced_id]),
                        'rule_info': rule_applied_info
                    }
                    forced_token_queue.appendleft(queued)
        
        generated_text_parts.append(candidate_data['token'])
        generated_text += candidate_data['token']
        
        if normalized_rules:
            for rule in normalized_rules:
                update_rule_state_with_token(rule, candidate_data['token'], candidate_data['probability'])
        
        token_payload = {
            "token": candidate_data['token'],
            "token_id": int(candidate_data['token_id']),
            "probability": candidate_data['probability'],
            "rank": candidate_data['rank'],
            "vocab_size": vocab_size,
            "top_alternatives": candidate_data['top_alternatives']
        }
        if rule_applied_info:
            token_payload['rule_applied'] = rule_applied_info
        
        yield token_payload
        
        if candidate_data['token_id'] in eos_token_ids:
            print(f"EOS token detected: {candidate_data['token']} (id: {candidate_data['token_id']})")
            break
        
        generated_ids = torch.cat([generated_ids, torch.tensor([[candidate_data['token_id']]], device=device)], dim=1)
    
    # Clean up memory after generation
    if device == 'cuda':
        del generated_ids
        torch.cuda.empty_cache()


@app.route('/')
def chat():
    """Chat interface route."""
    return render_template('chat.html')


@app.route('/chat/<conversation_id>')
def chat_with_id(conversation_id):
    """Chat interface route with specific conversation ID."""
    return render_template('chat.html', conversation_id=conversation_id)


@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get all conversations."""
    conversations_dir = 'conversations'
    if not os.path.exists(conversations_dir):
        return json.dumps({})
    
    conversations = {}
    for filename in os.listdir(conversations_dir):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(conversations_dir, filename), 'r', encoding='utf-8') as f:
                    conv = json.load(f)
                    conversations[conv['id']] = conv
            except:
                pass
    
    return json.dumps(conversations)


@app.route('/api/conversations/<conversation_id>', methods=['GET', 'PUT', 'DELETE'])
def manage_conversation(conversation_id):
    """Get, update, or delete a specific conversation."""
    conversations_dir = 'conversations'
    filepath = os.path.join(conversations_dir, f'{conversation_id}.json')
    
    if request.method == 'GET':
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.dumps(json.load(f))
        return json.dumps(None), 404
    
    elif request.method == 'PUT':
        if not os.path.exists(conversations_dir):
            os.makedirs(conversations_dir)
            print(f"Created conversations directory: {conversations_dir}")
        
        conversation = request.get_json()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
        
        print(f"Saved conversation: {conversation_id} - {conversation.get('title', 'Untitled')}")
        return json.dumps({'success': True})
    
    elif request.method == 'DELETE':
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Deleted conversation: {conversation_id}")
            return json.dumps({'success': True})
        return json.dumps({'success': False}), 404


@app.route('/api/search-tokens', methods=['POST'])
def search_tokens():
    """Search for tokens in the tokenizer vocabulary."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        model_name = data.get('model', DEFAULT_MODEL)
        context = data.get('context', [])  # Conversation history
        prefix_tokens = data.get('prefix_tokens', [])  # Tokens before injection point
        
        print(f"Token search request: query='{query}', model={model_name}")
        
        if not query:
            return json.dumps([])
        
        # Load tokenizer (model only if needed later)
        tokenizer = get_tokenizer(model_name, log_details=False)
        model = None
        device = 'cpu'
        
        # Search for matching tokens
        query_lower = query.lower()
        
        # Get vocabulary
        vocab = tokenizer.get_vocab()
        print(f"Vocabulary size: {len(vocab)}")
        
        # Find tokens that contain the query (case-insensitive)
        matches = []
        seen_tokens = set()  # Avoid duplicates
        
        for token_str, token_id in vocab.items():
            # Decode the token properly
            decoded = tokenizer.decode([token_id])
            
            # Skip if we've seen this decoded token already
            if decoded in seen_tokens:
                continue
            
            # Search in both raw and decoded token
            if query_lower in token_str.lower() or query_lower in decoded.lower():
                matches.append({
                    'token': decoded,
                    'token_id': token_id,
                    'raw': token_str
                })
                seen_tokens.add(decoded)
                
                if len(matches) >= 50:  # Get more results to filter
                    break
        
        # Calculate probabilities if context is provided
        probabilities = {}
        if (context or prefix_tokens) and model_name in MODEL_CACHE:
            import torch
            model, tokenizer, device = MODEL_CACHE[model_name]
            try:
                # Build the context
                if context and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                    formatted_context = tokenizer.apply_chat_template(
                        context,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    context_ids = tokenizer.encode(formatted_context, return_tensors="pt").to(device)
                else:
                    context_ids = torch.tensor([[]], dtype=torch.long).to(device)
                
                # Add prefix tokens if provided
                if prefix_tokens:
                    prefix_text = ''.join([t['token'] for t in prefix_tokens])
                    prefix_ids = tokenizer.encode(prefix_text, return_tensors="pt", add_special_tokens=False).to(device)
                    if context_ids.size(1) > 0:
                        all_ids = torch.cat([context_ids, prefix_ids], dim=1)
                    else:
                        all_ids = prefix_ids
                else:
                    all_ids = context_ids
                
                # Get logits at the current position
                if all_ids.size(1) > 0:
                    with torch.no_grad():
                        logits = model(all_ids).logits[:, -1, :]  # Get last position
                    
                    # Calculate probabilities for all tokens
                    all_probs = torch.softmax(logits[0], dim=-1)
                    
                    # Extract probabilities for matching tokens
                    for match in matches:
                        token_id = match['token_id']
                        probabilities[token_id] = all_probs[token_id].item()
                        
                    print(f"Calculated probabilities for {len(matches)} matching tokens")
                    
            except Exception as e:
                print(f"Error calculating probabilities: {e}")
                import traceback
                traceback.print_exc()
                # Continue without probabilities
        elif context or prefix_tokens:
            print("Skipping probability calculation for token search - model not loaded yet.")
        
        # Add probabilities to matches
        for match in matches:
            if match['token_id'] in probabilities:
                match['probability'] = probabilities[match['token_id']]
            else:
                match['probability'] = None
        
        # Sort by relevance and probability:
        # 1. Exact matches first
        # 2. Starts with query
        # 3. Then by probability (if available) or length
        def sort_key(x):
            token_lower = x['token'].lower()
            prob = x.get('probability', 0) or 0
            
            if token_lower == query_lower:
                return (0, -prob)  # Negative for descending order
            elif token_lower.startswith(query_lower):
                return (1, -prob)
            else:
                return (2, -prob if prob > 0 else len(x['token']))
        
        matches.sort(key=sort_key)
        
        print(f"Found {len(matches)} matches for '{query}', returning top 15")
        
        response_data = matches[:15]  # Return top 15
        return Response(json.dumps(response_data), mimetype='application/json')
        
    except Exception as e:
        print(f"Error searching tokens: {e}")
        import traceback
        traceback.print_exc()
        return Response(json.dumps([]), mimetype='application/json', status=500)


@app.route('/api/calculate-logprobs', methods=['POST'])
def calculate_logprobs():
    """Calculate logprobs for user input text."""
    import torch
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_name = data.get('model', DEFAULT_MODEL)
        context = data.get('context', [])  # Previous messages for context
        temperature = data.get('temperature', 1.0)
        
        if not text:
            return json.dumps({'tokens': []})
        
        # Load model
        model, tokenizer, device = load_model_and_tokenizer(model_name)
        
        # Build context (previous messages)
        if context and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            formatted_context = tokenizer.apply_chat_template(
                context,
                tokenize=False,
                add_generation_prompt=False
            )
            context_ids = tokenizer.encode(formatted_context, return_tensors="pt").to(device)
        else:
            context_ids = torch.tensor([[]], dtype=torch.long).to(device)
        
        # Tokenize the user's text
        text_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)
        
        # Combine context and text
        all_ids = torch.cat([context_ids, text_ids], dim=1) if context_ids.size(1) > 0 else text_ids
        
        # Get logits
        with torch.no_grad():
            logits = model(all_ids).logits
        
        # Calculate probabilities for each token in the text
        token_data = []
        context_len = context_ids.size(1)
        
        for i, token_id in enumerate(text_ids[0]):
            token_id = token_id.item()
            
            # Get logits for position that predicts this token
            if context_len + i > 0:
                current_logits = logits[0, context_len + i - 1]
                
                # Apply temperature scaling if not greedy
                if temperature != 0 and temperature >= 0.01:
                    current_logits = current_logits / temperature
                
                probabilities = torch.softmax(current_logits, dim=-1)
                
                prob = probabilities[token_id].item()
                
                # Calculate rank
                sorted_indices = torch.argsort(probabilities, descending=True)
                rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1
                
                # Get top 3 alternatives (will filter current token in UI to show 2)
                top_k_probs, top_k_indices = torch.topk(probabilities, 3)
                top_alternatives = [
                    {"token": tokenizer.decode([idx.item()]), "probability": p.item(), "rank": rank_idx + 1}
                    for rank_idx, (idx, p) in enumerate(zip(top_k_indices, top_k_probs))
                ]
            else:
                # First token has no previous context
                prob = None
                rank = None
                top_alternatives = []
            
            token_data.append({
                'token': tokenizer.decode([token_id]),
                'probability': prob,
                'rank': rank,
                'vocab_size': tokenizer.vocab_size,
                'top_alternatives': top_alternatives
            })
        
        return json.dumps({'tokens': token_data})
        
    except Exception as e:
        print(f"Error calculating logprobs: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({'tokens': []}), 500


@app.route('/api/analyze-difflens', methods=['POST'])
def analyze_difflens():
    """Analyze tokens with analysis model to compare probabilities."""
    import torch
    import math
    
    try:
        data = request.get_json()
        generation_model_name = data.get('generation_model')
        analysis_model_name = data.get('analysis_model')
        context = data.get('context', [])
        tokens = data.get('tokens', [])
        temperature = data.get('temperature', 1.0)
        
        if not analysis_model_name or not tokens:
            return jsonify({'token_data': []})
        
        # Load analysis model
        analysis_model, analysis_tokenizer, analysis_device = load_model_and_tokenizer(analysis_model_name)
        
        # Check if models are the same (can use token IDs directly)
        models_match = (generation_model_name == analysis_model_name)
        
        # Build assistant token IDs if provided AND models match (for perfectly matching generation)
        token_id_sequence = []
        can_use_token_ids = models_match
        
        if models_match:
            for t in tokens:
                token_id_value = t.get('token_id')
                if token_id_value is None:
                    can_use_token_ids = False
                    break
                try:
                    token_id_sequence.append(int(token_id_value))
                except (ValueError, TypeError):
                    can_use_token_ids = False
                    break
        
        # Always have text as fallback
        assistant_text = ''.join([t.get('token', '') for t in tokens])
        
        # Build context messages (mirror full conversation history prior to analyzed message)
        context_messages = []
        for msg in context:
            if not isinstance(msg, dict):
                continue
            role = msg.get('role')
            content = msg.get('content', '')
            if role is None:
                continue
            context_messages.append({
                'role': role,
                'content': content
            })
        
        # Format context
        if context_messages and hasattr(analysis_tokenizer, 'apply_chat_template') and analysis_tokenizer.chat_template:
            formatted_context = analysis_tokenizer.apply_chat_template(
                context_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_context = ''
        
        # Tokenize context (match generation behavior - include special tokens)
        if formatted_context:
            context_ids = analysis_tokenizer.encode(formatted_context, return_tensors="pt").to(analysis_device)
        else:
            context_ids = torch.tensor([[]], dtype=torch.long).to(analysis_device)
        
        # Only use token IDs if models match (same tokenizer)
        if can_use_token_ids and token_id_sequence:
            assistant_ids = torch.tensor([token_id_sequence], dtype=torch.long).to(analysis_device)
        else:
            # Re-tokenize with analysis model's tokenizer (for cross-model comparison)
            assistant_ids = analysis_tokenizer.encode(assistant_text, return_tensors="pt", add_special_tokens=False).to(analysis_device)
        full_ids = torch.cat([context_ids, assistant_ids], dim=1) if context_ids.size(1) > 0 else assistant_ids
        
        # Get logits
        with torch.no_grad():
            logits = analysis_model(full_ids).logits
        
        # Process each generation token
        token_data = []
        context_len = context_ids.size(1)
        vocab_size = analysis_tokenizer.vocab_size
        
        # When models match, we can do 1:1 token comparison
        # When models differ, tokenization may differ - handle gracefully
        analysis_token_count = assistant_ids.size(1)
        
        for i in range(len(tokens)):
            gen_token_info = tokens[i]
            gen_token = gen_token_info.get('token', '')
            gen_prob = gen_token_info.get('gen_prob', 0)
            gen_rank = gen_token_info.get('gen_rank', None)
            gen_top_alternatives = gen_token_info.get('gen_top_alternatives', [])
            
            # Handle token comparison based on whether models match
            if models_match and i < analysis_token_count:
                # Perfect alignment - same tokenizer, use token IDs directly
                token_id = assistant_ids[0, i].item()
                
                if context_len + i > 0:
                    # Get logits for position that predicts this token
                    current_logits = logits[0, context_len + i - 1]
                    
                    # Apply temperature if not greedy
                    if temperature != 0 and temperature >= 0.01:
                        current_logits = current_logits / temperature
                    
                    probabilities = torch.softmax(current_logits, dim=-1)
                    analysis_prob = probabilities[token_id].item()
                    
                    # Calculate rank
                    sorted_indices = torch.argsort(probabilities, descending=True)
                    analysis_rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1
                    
                    # Get top 3 (will filter current token in UI to show 2)
                    top_k_probs, top_k_indices = torch.topk(probabilities, min(3, vocab_size))
                    analysis_top_alternatives = [
                        {
                            'token': analysis_tokenizer.decode([int(idx)]),
                            'probability': float(prob),
                            'rank': rank_idx + 1
                        }
                        for rank_idx, (prob, idx) in enumerate(zip(top_k_probs.cpu().tolist(), top_k_indices.cpu().tolist()))
                    ]
                    
                    # Calculate differences (as percentages)
                    prob_diff = (gen_prob - analysis_prob) * 100  # Percentage point difference
                    rank_diff = (analysis_rank - gen_rank) if gen_rank else 0
                    
                    token_data.append({
                        'token': gen_token,
                        'gen_prob': float(gen_prob),
                        'gen_rank': int(gen_rank) if gen_rank else None,
                        'gen_top_alternatives': gen_top_alternatives,
                        'analysis_prob': float(analysis_prob),
                        'analysis_rank': int(analysis_rank),
                        'analysis_top_alternatives': analysis_top_alternatives,
                        'prob_diff': float(prob_diff),
                        'rank_diff': int(rank_diff)
                    })
                else:
                    # First token
                    token_data.append({
                        'token': gen_token,
                        'gen_prob': float(gen_prob),
                        'gen_rank': int(gen_rank) if gen_rank else None,
                        'gen_top_alternatives': gen_top_alternatives,
                        'analysis_prob': 0,
                        'analysis_rank': None,
                        'analysis_top_alternatives': [],
                        'prob_diff': 0,
                        'rank_diff': 0
                    })
            elif not models_match:
                # Cross-model comparison with different tokenizers
                # Approximate matching: find where this generation token text appears in analysis tokenization
                
                # Build text up to this point
                text_before = ''.join([t.get('token', '') for t in tokens[:i]])
                text_up_to_here = text_before + gen_token
                
                # Encode the full text to find position
                if text_up_to_here:
                    test_ids = analysis_tokenizer.encode(text_up_to_here, return_tensors="pt", add_special_tokens=False).to(analysis_device)
                    # The last token position in test_ids represents the end of our generation token
                    analysis_pos = test_ids.size(1) - 1
                    
                    if analysis_pos >= 0 and analysis_pos < analysis_token_count:
                        # Get the token ID at this position
                        token_id = assistant_ids[0, analysis_pos].item()
                        
                        # Get logits that predict this position
                        if context_len + analysis_pos > 0:
                            current_logits = logits[0, context_len + analysis_pos - 1]
                            
                            # Apply temperature if not greedy
                            if temperature != 0 and temperature >= 0.01:
                                current_logits = current_logits / temperature
                            
                            probabilities = torch.softmax(current_logits, dim=-1)
                            analysis_prob = probabilities[token_id].item()
                            
                            # Calculate rank
                            sorted_indices = torch.argsort(probabilities, descending=True)
                            analysis_rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1
                            
                            # Get top 3 alternatives
                            top_k_probs, top_k_indices = torch.topk(probabilities, min(3, vocab_size))
                            analysis_top_alternatives = [
                                {
                                    'token': analysis_tokenizer.decode([int(idx)]),
                                    'probability': float(prob),
                                    'rank': rank_idx + 1
                                }
                                for rank_idx, (prob, idx) in enumerate(zip(top_k_probs.cpu().tolist(), top_k_indices.cpu().tolist()))
                            ]
                            
                            # Calculate differences
                            prob_diff = (gen_prob - analysis_prob) * 100
                            rank_diff = (analysis_rank - gen_rank) if gen_rank else 0
                            
                            token_data.append({
                                'token': gen_token,
                                'gen_prob': float(gen_prob),
                                'gen_rank': int(gen_rank) if gen_rank else None,
                                'gen_top_alternatives': gen_top_alternatives,
                                'analysis_prob': float(analysis_prob),
                                'analysis_rank': int(analysis_rank),
                                'analysis_top_alternatives': analysis_top_alternatives,
                                'prob_diff': float(prob_diff),
                                'rank_diff': int(rank_diff)
                            })
                            continue
                
                # Fallback: can't properly analyze this token
                token_data.append({
                    'token': gen_token,
                    'gen_prob': float(gen_prob),
                    'gen_rank': int(gen_rank) if gen_rank else None,
                    'gen_top_alternatives': gen_top_alternatives,
                    'analysis_prob': 0,
                    'analysis_rank': None,
                    'analysis_top_alternatives': [],
                    'prob_diff': 0,
                    'rank_diff': 0
                })
            else:
                # Token index out of bounds (shouldn't happen with models_match)
                token_data.append({
                    'token': gen_token,
                    'gen_prob': float(gen_prob),
                    'gen_rank': int(gen_rank) if gen_rank else None,
                    'gen_top_alternatives': gen_top_alternatives,
                    'analysis_prob': 0,
                    'analysis_rank': None,
                    'analysis_top_alternatives': [],
                    'prob_diff': 0,
                    'rank_diff': 0
                })
        
        # Clean up memory after analysis
        if analysis_device == 'cuda':
            torch.cuda.empty_cache()
        
        return jsonify({
            'token_data': token_data,
            'generation_model': generation_model_name,
            'analysis_model': analysis_model_name
        })
        
    except RuntimeError as e:
        # Handle CUDA OOM errors specifically
        if 'out of memory' in str(e).lower():
            print(f"CUDA OOM Error in DiffLens: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Cleared CUDA cache after OOM error")
            return jsonify({'token_data': [], 'error': 'GPU out of memory. Try analyzing fewer tokens or using a smaller model.'}), 500
        else:
            print(f"Runtime error in DiffLens analysis: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'token_data': []}), 500
    except Exception as e:
        print(f"Error in DiffLens analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'token_data': []}), 500


@app.route('/api/logit-lens', methods=['POST'])
def logit_lens():
    """Perform logit lens analysis on a specific token position."""
    import torch
    
    try:
        data = request.get_json()
        model_name = data.get('model', DEFAULT_MODEL)
        context = data.get('context', [])  # Previous messages
        context_tokens = data.get('context_tokens', [])  # Tokens before the window
        window_tokens = data.get('window_tokens', [])  # The tokens in the window
        top_k = data.get('top_k', 20)  # Top K tokens to track across layers
        
        if not window_tokens or len(window_tokens) > 20:
            return jsonify({'error': 'Invalid window tokens'}), 400
        
        # Load model
        model, tokenizer, device = load_model_and_tokenizer(model_name)
        
        # Build context from chat history
        if context and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            formatted_context = tokenizer.apply_chat_template(
                context,
                tokenize=False,
                add_generation_prompt=True
            )
            context_ids = tokenizer.encode(formatted_context, return_tensors="pt").to(device)
        else:
            context_ids = torch.tensor([[]], dtype=torch.long).to(device)
        
        # Tokenize the context tokens (before the window)
        if context_tokens:
            context_text = ''.join([t.get('token', '') for t in context_tokens])
            context_text_ids = tokenizer.encode(context_text, return_tensors="pt", add_special_tokens=False).to(device)
        else:
            context_text_ids = torch.tensor([[]], dtype=torch.long).to(device)
        
        # Combine chat context and text context
        if context_ids.size(1) > 0 and context_text_ids.size(1) > 0:
            base_ids = torch.cat([context_ids, context_text_ids], dim=1)
        elif context_ids.size(1) > 0:
            base_ids = context_ids
        elif context_text_ids.size(1) > 0:
            base_ids = context_text_ids
        else:
            base_ids = torch.tensor([[]], dtype=torch.long).to(device)
        
        # Get number of layers
        num_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
        
        if num_layers == 0:
            # Try alternative model structures
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                num_layers = len(model.transformer.h)
            else:
                return jsonify({'error': 'Could not determine model layers'}), 500
        
        # Get model components
        if hasattr(model, 'model'):
            embed_tokens = model.model.embed_tokens if hasattr(model.model, 'embed_tokens') else None
            norm = model.model.norm if hasattr(model.model, 'norm') else None
            lm_head = model.lm_head if hasattr(model, 'lm_head') else None
            layers = model.model.layers if hasattr(model.model, 'layers') else None
        elif hasattr(model, 'transformer'):
            embed_tokens = model.transformer.wte if hasattr(model.transformer, 'wte') else None
            norm = model.transformer.ln_f if hasattr(model.transformer, 'ln_f') else None
            lm_head = model.lm_head if hasattr(model, 'lm_head') else None
            layers = model.transformer.h if hasattr(model.transformer, 'h') else None
        else:
            return jsonify({'error': 'Could not access model layers'}), 500
        
        positions_data = []
        
        # Analyze each position in the window
        for pos_idx in range(len(window_tokens)):
            # Build input: base_ids + tokens before this position
            if pos_idx == 0:
                position_ids = base_ids
            else:
                tokens_before = window_tokens[:pos_idx]
                text_before = ''.join([t.get('token', '') for t in tokens_before])
                tokens_before_ids = tokenizer.encode(text_before, return_tensors="pt", add_special_tokens=False).to(device)
                
                if base_ids.size(1) > 0:
                    position_ids = torch.cat([base_ids, tokens_before_ids], dim=1)
                else:
                    position_ids = tokens_before_ids
            
            if position_ids.size(1) == 0:
                continue
            
            # Capture layer outputs
            layer_outputs = []
            hooks = []
            
            def create_hook(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    layer_outputs.append((layer_idx, hidden_states.detach()))
                return hook
            
            if layers:
                for i, layer in enumerate(layers):
                    hook = layer.register_forward_hook(create_hook(i))
                    hooks.append(hook)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(position_ids, output_hidden_states=True)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            target_position = position_ids.size(1) - 1
            
            # Get predictions at each layer
            layer_predictions = []
            
            for layer_idx, hidden_states in layer_outputs:
                hidden_state = hidden_states[0, target_position, :]
                
                # Apply norm
                if norm is not None:
                    hidden_state = norm(hidden_state)
                
                # Get logits
                if lm_head is not None:
                    logits = lm_head(hidden_state)
                elif embed_tokens is not None:
                    logits = torch.matmul(hidden_state, embed_tokens.weight.T)
                else:
                    continue
                
                # Get probabilities and top k
                probs = torch.softmax(logits, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
                
                top_predictions = []
                for prob, idx in zip(top_k_probs, top_k_indices):
                    token_str = tokenizer.decode([idx.item()])
                    top_predictions.append({
                        'token': token_str,
                        'probability': prob.item(),
                        'token_id': idx.item()
                    })
                
                layer_predictions.append({
                    'layer': layer_idx,
                    'predictions': top_predictions
                })
            
            # Final layer (actual model output)
            final_logits = outputs.logits[0, target_position, :]
            final_probs = torch.softmax(final_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(final_probs, min(top_k, final_probs.size(-1)))
            
            final_predictions = []
            for prob, idx in zip(top_k_probs, top_k_indices):
                token_str = tokenizer.decode([idx.item()])
                final_predictions.append({
                    'token': token_str,
                    'probability': prob.item(),
                    'token_id': idx.item()
                })
            
            layer_predictions.append({
                'layer': num_layers,
                'predictions': final_predictions
            })
            
            positions_data.append({
                'position': pos_idx,
                'layer_predictions': layer_predictions
            })
        
        # Clean up memory after analysis
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        return jsonify({
            'num_layers': num_layers + 1,
            'positions': positions_data
        })
        
    except RuntimeError as e:
        # Handle CUDA OOM errors specifically
        if 'out of memory' in str(e).lower():
            print(f"CUDA OOM Error in logit lens: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Cleared CUDA cache after OOM error")
            return jsonify({'error': 'GPU out of memory. Try analyzing fewer tokens at once.'}), 500
        else:
            print(f"Runtime error in logit lens analysis: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"Error in logit lens analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/stream', methods=['POST'])
def stream():
    """SSE endpoint for streaming token generation."""
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        model_name = data.get('model', DEFAULT_MODEL)
        prefill = data.get('prefill', None)
        temperature = data.get('temperature', 1.0)
        show_prompt_tokens = data.get('show_prompt_tokens', False)
        generation_rules = data.get('rules', [])
        
        if not messages:
            return Response("data: " + json.dumps({"type": "error", "message": "No messages provided"}) + "\n\n", mimetype='text/event-stream')
        
        def generate():
            try:
                # Load model
                model, tokenizer, device = load_model_and_tokenizer(model_name)
                
                # Generate tokens with streaming
                first_token = True
                for token_data in generate_streaming_tokens(
                    model,
                    tokenizer,
                    device,
                    messages,
                    temperature=temperature,
                    prefill=prefill,
                    show_prompt_tokens=show_prompt_tokens,
                    rules=generation_rules,
                    model_name=model_name
                ):
                    # Mark first token so frontend can show special context
                    if first_token and token_data.get('type') != 'prompt_token':
                        token_data['is_first_token'] = True
                        first_token = False
                    
                    # Send token data as SSE immediately (no buffering)
                    yield f"data: {json.dumps({'type': 'token', **token_data})}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except RuntimeError as e:
                # Handle CUDA OOM errors specifically
                if 'out of memory' in str(e).lower():
                    print(f"CUDA OOM Error: {e}")
                    import torch
                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("Cleared CUDA cache after OOM error")
                        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
                    yield f"data: {json.dumps({'type': 'error', 'message': 'GPU out of memory. Try using a smaller model or shorter context.'})}\n\n"
                else:
                    print(f"Runtime error during streaming: {e}")
                    import traceback
                    traceback.print_exc()
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            except Exception as e:
                print(f"Error during streaming: {e}")
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        response = Response(stream_with_context(generate()), mimetype='text/event-stream')
        # Disable buffering for immediate streaming
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Accel-Buffering'] = 'no'
        return response
    
    except Exception as e:
        print(f"Error parsing request: {e}")
        return Response("data: " + json.dumps({"type": "error", "message": f"Invalid request: {str(e)}"}) + "\n\n", mimetype='text/event-stream')


@app.route('/old')
def index():
    """Main route for token probability visualization."""
    # Parse request parameters
    mode = request.args.get('mode', 'file')  # 'file' or 'direct'
    selected_file = request.args.get('file')
    prob_threshold = request.args.get('prob_threshold', default=DEFAULT_PROB_THRESHOLD, type=float)
    threshold_mode = request.args.get('threshold_mode', default='probability', type=str)
    rank_threshold = request.args.get('rank_threshold', default=DEFAULT_RANK_THRESHOLD, type=int)
    max_rank_for_scaling = request.args.get('max_rank_for_scaling', default=DEFAULT_MAX_RANK_SCALING, type=int)
    min_prob_for_scaling = request.args.get('min_prob_for_scaling', default=DEFAULT_MIN_PROB_SCALING, type=float)
    
    # Initialize variables
    token_data = []
    lowest_probs = []
    prompt_text = None
    error = None
    token_input_list = []
    vocab_size = None
    model_name = None
    
    # Scan for available files
    available_files, scan_error = scan_available_files()
    if scan_error:
        error = scan_error
    
    # Check if we have direct analysis data from session
    if mode == 'direct' and 'analysis_data' in session:
        analysis_data = session.get('analysis_data')
        prompt_text = analysis_data.get('prompt')
        token_input_list = analysis_data.get('tokens', [])
        vocab_size = analysis_data.get('vocab_size')
        model_name = analysis_data.get('model_name')
    elif mode == 'file' or not session.get('analysis_data'):
        # File mode - existing logic
        if not error:
            file_to_load, selected_file = determine_file_to_load(available_files, selected_file)
            
            if not file_to_load and selected_file:
                error = f"Selected file '{selected_file}' not found in {OUTPUT_DATA_DIR}."
            elif not file_to_load and not available_files:
                error = f"No JSON files found in {OUTPUT_DATA_DIR}. Run probability_monitor.py first or use New Analysis."
            elif not file_to_load:
                error = None  # Allow showing the new analysis form
            
            # Load and process data if file is available
            if not error and file_to_load:
                prompt_text, token_input_list, vocab_size, model_name, load_error = load_token_data(file_to_load)
                
                if load_error:
                    error = load_error
    
    # Process data if we have token input (from either file or direct analysis)
    if token_input_list and vocab_size and not error:
        # Process token data
        token_data = process_token_data(token_input_list, vocab_size, threshold_mode, max_rank_for_scaling, min_prob_for_scaling)
        
        # Apply threshold filtering
        token_data, lowest_probs = apply_threshold_filtering(
            token_data, threshold_mode, prob_threshold, rank_threshold
        )

    return render_template('index.html', 
                          token_data=token_data, 
                          error=error,
                          prompt_text=prompt_text,
                          lowest_probs=lowest_probs,
                          available_files=available_files,
                          selected_file=selected_file,
                          prob_threshold=prob_threshold,
                          threshold_mode=threshold_mode,
                          rank_threshold=rank_threshold,
                          max_rank_for_scaling=max_rank_for_scaling,
                          min_prob_for_scaling=min_prob_for_scaling,
                          mode=mode,
                          model_name=model_name,
                          default_model=DEFAULT_MODEL)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Process direct prompt and output analysis."""
    try:
        prompt = request.form.get('prompt', '').strip()
        output = request.form.get('output', '').strip()
        model_name = request.form.get('model_name', DEFAULT_MODEL).strip()
        custom_filename = request.form.get('filename', '').strip()
        
        # Validation
        if not prompt:
            flash('Error: Prompt cannot be empty', 'error')
            return redirect(url_for('index'))
        
        if not output:
            flash('Error: Output cannot be empty', 'error')
            return redirect(url_for('index'))
        
        # Load model and calculate probabilities
        print(f'Loading model {model_name}...')
        model, tokenizer, device = load_model_and_tokenizer(model_name)
        
        print('Calculating token probabilities...')
        token_input_list, vocab_size = calculate_token_probabilities(model, tokenizer, device, prompt, output)
        
        # Use custom filename or generate timestamp-based one
        if custom_filename:
            # Sanitize filename
            safe_filename = re.sub(r'[^\w\-_]', '_', custom_filename)
            json_filename = f"{safe_filename}.json"
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"direct_analysis_{timestamp}.json"
        
        output_json_path = os.path.join(OUTPUT_DATA_DIR, json_filename)
        
        if not os.path.exists(OUTPUT_DATA_DIR):
            os.makedirs(OUTPUT_DATA_DIR)
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            output_container = {
                "prompt": prompt,
                "vocab_size": vocab_size,
                "model_name": model_name,
                "tokens": token_input_list
            }
            json.dump(output_container, f, ensure_ascii=False, indent=2)
        
        # Store in session for display
        session['analysis_data'] = {
            'prompt': prompt,
            'vocab_size': vocab_size,
            'model_name': model_name,
            'tokens': token_input_list
        }
        
        flash(f'Analysis complete! Saved as {json_filename}', 'success')
        return redirect(url_for('index', mode='direct'))
        
    except Exception as e:
        flash(f'Error during analysis: {str(e)}', 'error')
        return redirect(url_for('index'))


if __name__ == '__main__':
    # Enable threaded mode for better streaming performance
    app.run(debug=True, port=5001, threaded=True)