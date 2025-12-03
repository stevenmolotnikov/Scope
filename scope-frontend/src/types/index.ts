// ============================================================================
// Core Token Types
// ============================================================================

export interface TokenAlternative {
  token: string;
  probability: number;
  rank: number;
}

export interface RuleAppliedInfo {
  id: string;
  name: string;
  action: string;
  reason: string;
  original_token?: string;
  resampling_chain?: string[];
}

export interface TokenDiffData {
  analysis_prob: number;
  analysis_rank: number | null;
  analysis_top_alternatives: TokenAlternative[];
  prob_diff: number;
  rank_diff: number;
}

export interface Token {
  token: string;
  token_id: number;
  probability: number | null;
  rank: number | null;
  vocab_size: number;
  top_alternatives: TokenAlternative[];
  is_prefill_token?: boolean;
  is_prompt_token?: boolean;
  is_first_token?: boolean;
  rule_applied?: RuleAppliedInfo;
  diff_data?: TokenDiffData;
}

export interface ProcessedToken extends Token {
  color: string;
  display_token: string;
  token_class: string;
  original_index: number;
  output_token_index: number;
  is_lowest_prob?: boolean;
}

// ============================================================================
// Message Types
// ============================================================================

export type MessageRole = 'user' | 'assistant' | 'system';

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  tokens?: Token[];
  parentId: string | null;
  childrenIds: string[];
  activeChildIndex: number;
  timestamp?: number;
}

export type MessageTree = Record<string, Message>;

// ============================================================================
// Conversation Types
// ============================================================================

export interface Conversation {
  id: string;
  title: string;
  model: string;
  temperature: number;
  systemPrompt: string;
  messageTree: MessageTree;
  currentLeafId: string | null;
  createdAt?: number;
  updatedAt?: number;
}

export type ConversationsMap = Record<string, Conversation>;

// ============================================================================
// Generation Rule Types
// ============================================================================

export type RuleCriteriaType = 
  | 'probability_below'
  | 'consecutive_probability_below'
  | 'text_match';

export type RuleActionType = 
  | 'resample_same'
  | 'resample_other_model'
  | 'replace_text';

export type TextMatchType = 'contains' | 'exact' | 'regex';

export interface RuleCriteria {
  type: RuleCriteriaType;
  threshold?: number;
  count?: number;
  value?: string;
  window?: number;
  match_type?: TextMatchType;
}

export interface RuleAction {
  type: RuleActionType;
  strategy?: 'sample' | 'greedy';
  top_k?: number;
  max_attempts?: number;
  text?: string;
  model?: string;
  temperature?: number;
}

export interface GenerationRule {
  id: string;
  name: string;
  enabled: boolean;
  criteria: RuleCriteria;
  action: RuleAction;
}

// ============================================================================
// Sampling Settings
// ============================================================================

export interface SamplingSettings {
  top_k: number;
  top_p: number;
}

// ============================================================================
// API Request/Response Types
// ============================================================================

export interface StreamRequest {
  messages: Array<{ role: MessageRole; content: string }>;
  model: string;
  prefill?: string;
  temperature: number;
  show_prompt_tokens?: boolean;
  rules?: GenerationRule[];
}

export interface StreamTokenEvent {
  type: 'token';
  token: string;
  token_id: number;
  probability: number;
  rank: number;
  vocab_size: number;
  top_alternatives: TokenAlternative[];
  is_prefill_token?: boolean;
  is_prompt_token?: boolean;
  is_first_token?: boolean;
  rule_applied?: RuleAppliedInfo;
}

export interface StreamDoneEvent {
  type: 'done';
}

export interface StreamErrorEvent {
  type: 'error';
  message: string;
}

export type StreamEvent = StreamTokenEvent | StreamDoneEvent | StreamErrorEvent;

// ============================================================================
// DiffLens Types
// ============================================================================

export interface DiffLensRequest {
  generation_model: string;
  analysis_model: string;
  context: Array<{ role: MessageRole; content: string }>;
  tokens: Array<{
    token: string;
    token_id?: number;
    gen_prob: number;
    gen_rank: number | null;
    gen_top_alternatives: TokenAlternative[];
  }>;
  temperature: number;
}

export interface DiffLensTokenData {
  token: string;
  gen_prob: number;
  gen_rank: number | null;
  gen_top_alternatives: TokenAlternative[];
  analysis_prob: number;
  analysis_rank: number | null;
  analysis_top_alternatives: TokenAlternative[];
  prob_diff: number;
  rank_diff: number;
}

export interface DiffLensResponse {
  token_data: DiffLensTokenData[];
  generation_model: string;
  analysis_model: string;
  error?: string;
}

// ============================================================================
// Logit Lens Types
// ============================================================================

export interface LogitLensRequest {
  model: string;
  context: Array<{ role: MessageRole; content: string }>;
  context_tokens: Token[];
  window_tokens: Token[];
  top_k?: number;
}

export interface LogitLensPrediction {
  token: string;
  probability: number;
  token_id: number;
}

export interface LogitLensLayerData {
  layer: number;
  predictions: LogitLensPrediction[];
}

export interface LogitLensPositionData {
  position: number;
  layer_predictions: LogitLensLayerData[];
}

export interface LogitLensResponse {
  num_layers: number;
  positions: LogitLensPositionData[];
  error?: string;
}

// ============================================================================
// Token Search Types
// ============================================================================

export interface TokenSearchRequest {
  query: string;
  model: string;
  context?: Array<{ role: MessageRole; content: string }>;
  prefix_tokens?: Token[];
}

export interface TokenSearchResult {
  token: string;
  token_id: number;
  raw: string;
  probability?: number | null;
}

// ============================================================================
// UI State Types
// ============================================================================

export type ViewMode = 'token' | 'text' | 'diff';
export type HighlightMode = 'probability' | 'rank';

export interface SelectedTokenInfo {
  token: Token;
  messageId: string;
  tokenIndex: number;
  element?: HTMLElement;
}

// ============================================================================
// Model Configuration
// ============================================================================

export interface ModelOption {
  value: string;
  label: string;
}

export const AVAILABLE_MODELS: ModelOption[] = [
  { value: 'meta-llama/Llama-3.1-8B-Instruct', label: 'Llama-3.1-8B-Instruct' },
  { value: 'google/gemma-2-2b-it', label: 'Gemma-2-2B-IT' },
  { value: 'google/gemma-3-1b-it', label: 'Gemma-3-1B-IT' },
];

export const DEFAULT_MODEL = 'meta-llama/Llama-3.1-8B-Instruct';
export const DEFAULT_TEMPERATURE = 1.0;

