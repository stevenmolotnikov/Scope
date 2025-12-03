'use client';

import { useCallback, useRef } from 'react';
import { api, streamSSE } from '@/lib/api';
import { useConversationStore } from '@/stores/conversationStore';
import { useUIStore } from '@/stores/uiStore';
import type { Token, StreamEvent, MessageRole, Message, TokenDiffData } from '@/types';

interface StreamingOptions {
  prefill?: string;
  diffModel?: string;
  onToken?: (token: Token) => void;
  onComplete?: () => void;
  onError?: (error: string) => void;
}

export function useStreamingGeneration() {
  const tokensRef = useRef<Token[]>([]);
  const messageIdRef = useRef<string | null>(null);

  const {
    getCurrentConversation,
    getConversationHistory,
    addUserMessage,
    addAssistantMessage,
    updateMessageTokens,
    setGenerating,
    stopGeneration,
    generationRules,
    saveConversation,
  } = useConversationStore();

  const { prefillEnabled, prefillText, setPrefillText, diffLensEnabled, diffLensModel } = useUIStore();

  /**
   * Run DiffLens analysis after generation
   */
  const runDiffLens = useCallback(
    async (
      tokens: Token[],
      generationModel: string,
      analysisModel: string,
      context: Array<{ role: MessageRole; content: string }>,
      temperature: number,
      assistantMessageId: string
    ) => {
      if (tokens.length === 0) return;

      try {
        const response = await api.analyzeDifflens({
          generation_model: generationModel,
          analysis_model: analysisModel,
          context,
          tokens: tokens.map(t => ({
            token: t.token,
            token_id: t.token_id,
            gen_prob: t.probability ?? 0,
            gen_rank: t.rank,
            gen_top_alternatives: t.top_alternatives,
          })),
          temperature,
        });

        if (response.error) {
          console.error('DiffLens error:', response.error);
          return;
        }

        // Merge diff data into tokens
        const tokensWithDiff = tokens.map((token, idx) => {
          const diffData = response.token_data[idx];
          if (!diffData) return token;

          const diff: TokenDiffData = {
            analysis_prob: diffData.analysis_prob,
            analysis_rank: diffData.analysis_rank,
            analysis_top_alternatives: diffData.analysis_top_alternatives,
            prob_diff: diffData.prob_diff,
            rank_diff: diffData.rank_diff,
          };

          return { ...token, diff_data: diff };
        });

        // Update tokens with diff data
        updateMessageTokens(assistantMessageId, tokensWithDiff);
      } catch (error) {
        console.error('DiffLens analysis failed:', error);
      }
    },
    [updateMessageTokens]
  );

  /**
   * Send a message and stream the response
   */
  const sendMessage = useCallback(
    async (content: string, options: StreamingOptions = {}) => {
      const conversation = getCurrentConversation();
      if (!conversation) {
        options.onError?.('No conversation selected');
        return;
      }

      // Add user message to tree
      const userMessageId = addUserMessage(content);
      if (!userMessageId) {
        options.onError?.('Failed to add message');
        return;
      }

      // Add assistant message placeholder
      const assistantMessageId = addAssistantMessage(userMessageId);
      if (!assistantMessageId) {
        options.onError?.('Failed to create assistant message');
        return;
      }

      messageIdRef.current = assistantMessageId;
      tokensRef.current = [];

      // Create abort controller
      const abortController = new AbortController();
      setGenerating(true, abortController);

      try {
        // Build message history
        const history = getConversationHistory();
        const messages = history.map((msg) => ({
          role: msg.role as MessageRole,
          content: msg.content,
        }));

        // Determine prefill
        const prefill = options.prefill ?? (prefillEnabled ? prefillText : undefined);

        // Determine diff model
        const diffModel = options.diffModel ?? (diffLensEnabled && diffLensModel ? diffLensModel : undefined);

        // Make streaming request
        const response = await api.streamGeneration(
          {
            messages,
            model: conversation.model,
            temperature: conversation.temperature,
            prefill: prefill || undefined,
            rules: generationRules.length > 0 ? generationRules : undefined,
          },
          abortController.signal
        );

        // Process SSE stream
        for await (const event of streamSSE(response)) {
          if (abortController.signal.aborted) {
            break;
          }

          await processStreamEvent(event, assistantMessageId, options);
        }

        // Run DiffLens if enabled
        if (diffModel && tokensRef.current.length > 0 && !abortController.signal.aborted) {
          await runDiffLens(
            tokensRef.current,
            conversation.model,
            diffModel,
            messages,
            conversation.temperature,
            assistantMessageId
          );
        }

        // Clear prefill after successful generation
        if (prefillEnabled && prefillText) {
          setPrefillText('');
        }

        // Save conversation
        const updatedConv = getCurrentConversation();
        if (updatedConv) {
          saveConversation(updatedConv);
        }

        options.onComplete?.();
      } catch (error) {
        if (error instanceof Error && error.name === 'AbortError') {
          // User cancelled - save current progress
          const updatedConv = getCurrentConversation();
          if (updatedConv) {
            saveConversation(updatedConv);
          }
        } else {
          console.error('Streaming error:', error);
          options.onError?.(error instanceof Error ? error.message : 'Unknown error');
        }
      } finally {
        setGenerating(false, null);
        messageIdRef.current = null;
      }
    },
    [
      getCurrentConversation,
      getConversationHistory,
      addUserMessage,
      addAssistantMessage,
      setGenerating,
      generationRules,
      prefillEnabled,
      prefillText,
      diffLensEnabled,
      diffLensModel,
      setPrefillText,
      saveConversation,
      runDiffLens,
    ]
  );

  /**
   * Process a single stream event
   */
  const processStreamEvent = useCallback(
    async (
      event: StreamEvent,
      assistantMessageId: string,
      options: StreamingOptions
    ) => {
      switch (event.type) {
        case 'token': {
          const token: Token = {
            token: event.token,
            token_id: event.token_id,
            probability: event.probability,
            rank: event.rank,
            vocab_size: event.vocab_size,
            top_alternatives: event.top_alternatives,
            is_prefill_token: event.is_prefill_token,
            is_prompt_token: event.is_prompt_token,
            is_first_token: event.is_first_token,
            rule_applied: event.rule_applied,
          };

          // Skip prompt tokens from display
          if (token.is_prompt_token) {
            return;
          }

          tokensRef.current = [...tokensRef.current, token];
          updateMessageTokens(assistantMessageId, tokensRef.current);
          options.onToken?.(token);
          break;
        }

        case 'done':
          // Generation complete
          break;

        case 'error':
          options.onError?.(event.message);
          break;
      }
    },
    [updateMessageTokens]
  );

  /**
   * Regenerate from a specific message
   */
  const regenerateFrom = useCallback(
    async (messageId: string, options: StreamingOptions = {}) => {
      const conversation = getCurrentConversation();
      if (!conversation) {
        options.onError?.('No conversation selected');
        return;
      }

      const message = conversation.messageTree[messageId];
      if (!message || message.role !== 'user') {
        options.onError?.('Can only regenerate from user messages');
        return;
      }

      // Create new assistant message as sibling
      const assistantMessageId = addAssistantMessage(messageId);
      if (!assistantMessageId) {
        options.onError?.('Failed to create assistant message');
        return;
      }

      messageIdRef.current = assistantMessageId;
      tokensRef.current = [];

      const abortController = new AbortController();
      setGenerating(true, abortController);

      try {
        // Build history up to and including the user message
        const history: Array<{ role: MessageRole; content: string }> = [];

        if (conversation.systemPrompt) {
          history.push({ role: 'system', content: conversation.systemPrompt });
        }

        // Walk from root to the user message
        let currentId: string | null = messageId;
        const pathToMessage: string[] = [];

        while (currentId) {
          pathToMessage.unshift(currentId);
          const msg: Message | undefined = conversation.messageTree[currentId];
          if (!msg) break;
          currentId = msg.parentId;
        }

        for (const id of pathToMessage) {
          const msg = conversation.messageTree[id];
          if (msg && msg.role !== 'system' && msg.content) {
            history.push({ role: msg.role as MessageRole, content: msg.content });
          }
        }

        const prefill = options.prefill ?? (prefillEnabled ? prefillText : undefined);
        const diffModel = options.diffModel ?? (diffLensEnabled && diffLensModel ? diffLensModel : undefined);

        const response = await api.streamGeneration(
          {
            messages: history,
            model: conversation.model,
            temperature: conversation.temperature,
            prefill: prefill || undefined,
            rules: generationRules.length > 0 ? generationRules : undefined,
          },
          abortController.signal
        );

        for await (const event of streamSSE(response)) {
          if (abortController.signal.aborted) {
            break;
          }

          await processStreamEvent(event, assistantMessageId, options);
        }

        // Run DiffLens if enabled
        if (diffModel && tokensRef.current.length > 0 && !abortController.signal.aborted) {
          await runDiffLens(
            tokensRef.current,
            conversation.model,
            diffModel,
            history,
            conversation.temperature,
            assistantMessageId
          );
        }

        if (prefillEnabled && prefillText) {
          setPrefillText('');
        }

        const updatedConv = getCurrentConversation();
        if (updatedConv) {
          saveConversation(updatedConv);
        }

        options.onComplete?.();
      } catch (error) {
        if (error instanceof Error && error.name !== 'AbortError') {
          console.error('Regeneration error:', error);
          options.onError?.(error instanceof Error ? error.message : 'Unknown error');
        }
      } finally {
        setGenerating(false, null);
        messageIdRef.current = null;
      }
    },
    [
      getCurrentConversation,
      addAssistantMessage,
      setGenerating,
      generationRules,
      prefillEnabled,
      prefillText,
      diffLensEnabled,
      diffLensModel,
      setPrefillText,
      processStreamEvent,
      saveConversation,
      runDiffLens,
    ]
  );

  /**
   * Inject text at a position and regenerate
   */
  const injectAndRegenerate = useCallback(
    async (
      messageId: string,
      tokenIndex: number,
      injectedText: string,
      options: StreamingOptions = {}
    ) => {
      const conversation = getCurrentConversation();
      if (!conversation) {
        options.onError?.('No conversation selected');
        return;
      }

      const message = conversation.messageTree[messageId];
      if (!message || message.role !== 'assistant' || !message.tokens) {
        options.onError?.('Invalid message for injection');
        return;
      }

      // Get tokens up to injection point
      const prefixTokens = message.tokens.slice(0, tokenIndex);
      const prefixText = prefixTokens.map((t) => t.token).join('');
      const newPrefill = prefixText + injectedText;

      // Find the parent user message
      const parentId = message.parentId;
      if (!parentId) {
        options.onError?.('No parent message found');
        return;
      }

      // Regenerate with the injection as prefill
      await regenerateFrom(parentId, {
        ...options,
        prefill: newPrefill,
      });
    },
    [getCurrentConversation, regenerateFrom]
  );

  return {
    sendMessage,
    regenerateFrom,
    injectAndRegenerate,
    stopGeneration,
  };
}
