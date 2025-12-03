'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { useConversationStore } from '@/stores/conversationStore';
import { useUIStore } from '@/stores/uiStore';
import { useStreamingGeneration } from '@/hooks/useStreamingGeneration';
import { api } from '@/lib/api';
import type { TokenDiffData, MessageRole } from '@/types';

const AVAILABLE_MODELS = [
  'google/gemma-3-1b-it',
  'google/gemma-2-2b-it',
];

export function ChatInput() {
  const [inputText, setInputText] = useState('');
  const [isApplyingDiff, setIsApplyingDiff] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const {
    isGenerating,
    getCurrentConversation,
    getConversationHistory,
    updateMessageTokens,
    setModel,
    setTemperature,
    saveConversation,
  } = useConversationStore();

  const {
    prefillEnabled,
    prefillText,
    setPrefillText,
    diffLensEnabled,
    diffLensModel,
    setDiffLensModel,
  } = useUIStore();

  const { sendMessage, stopGeneration } = useStreamingGeneration();

  const conversation = getCurrentConversation();

  const handleSubmit = useCallback(() => {
    if (!inputText.trim() || isGenerating) return;

    sendMessage(inputText.trim(), {
      prefill: prefillEnabled && prefillText.trim() ? prefillText.trim() : undefined,
      diffModel: diffLensEnabled && diffLensModel ? diffLensModel : undefined,
      onError: (error) => {
        console.error('Generation error:', error);
      },
    });

    setInputText('');
    if (prefillEnabled) {
      setPrefillText('');
    }
  }, [inputText, prefillEnabled, prefillText, diffLensEnabled, diffLensModel, isGenerating, sendMessage, setPrefillText]);

  const handleApplyDiff = useCallback(async () => {
    if (!conversation || !diffLensModel || isApplyingDiff) return;

    // Find ALL assistant messages with tokens
    const messageTree = conversation.messageTree;
    const assistantMessages = Object.values(messageTree)
      .filter(m => m.role === 'assistant' && m.tokens && m.tokens.length > 0);
    
    if (assistantMessages.length === 0) {
      console.error('No assistant messages with tokens to analyze');
      return;
    }

    setIsApplyingDiff(true);

    try {
      // Process each assistant message
      for (const assistantMsg of assistantMessages) {
        if (!assistantMsg.tokens) continue;

        // Build context up to this message
        const contextMessages: Array<{ role: MessageRole; content: string }> = [];
        
        // Walk the tree to build context
        let currentId: string | null = assistantMsg.parentId;
        const pathToRoot: string[] = [];
        while (currentId) {
          pathToRoot.unshift(currentId);
          const msg = messageTree[currentId];
          if (!msg) break;
          currentId = msg.parentId;
        }
        
        for (const id of pathToRoot) {
          const msg = messageTree[id];
          if (msg && msg.content) {
            contextMessages.push({
              role: msg.role as MessageRole,
              content: msg.content,
            });
          }
        }

        const response = await api.analyzeDifflens({
          generation_model: conversation.model,
          analysis_model: diffLensModel,
          context: contextMessages,
          tokens: assistantMsg.tokens.map(t => ({
            token: t.token,
            token_id: t.token_id,
            gen_prob: t.probability ?? 0,
            gen_rank: t.rank,
            gen_top_alternatives: t.top_alternatives,
          })),
          temperature: conversation.temperature,
        });

        if (response.error) {
          console.error('Diff error for message:', assistantMsg.id, response.error);
          continue;
        }

        // Merge diff data into tokens
        const tokensWithDiff = assistantMsg.tokens.map((token, idx) => {
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

        updateMessageTokens(assistantMsg.id, tokensWithDiff);
      }

      // Save conversation to persist diff data
      const updatedConv = getCurrentConversation();
      if (updatedConv) {
        saveConversation(updatedConv);
      }
    } catch (error) {
      console.error('Failed to apply diff:', error);
    } finally {
      setIsApplyingDiff(false);
    }
  }, [conversation, diffLensModel, isApplyingDiff, updateMessageTokens, getCurrentConversation, saveConversation]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
    }
  }, [inputText]);

  if (!conversation) return null;

  return (
    <div style={{
      borderTop: '1px solid #e5e5e5',
      background: '#fff',
      padding: '16px 24px',
    }}>
      {/* Prefill row (only when enabled) */}
      {prefillEnabled && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginBottom: '12px',
        }}>
          <label style={{ fontSize: '12px', color: '#666', fontWeight: 500, whiteSpace: 'nowrap' }}>
            Prefill:
          </label>
          <input
            type="text"
            value={prefillText}
            onChange={(e) => setPrefillText(e.target.value)}
            placeholder="Start response with..."
            className="font-mono"
            style={{
              flex: 1,
              padding: '6px 8px',
              border: '1px solid #ddd',
              borderRadius: '4px',
              fontSize: '12px',
              background: '#fff',
            }}
          />
        </div>
      )}

      {/* Diff row (only when enabled) */}
      {diffLensEnabled && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginBottom: '12px',
        }}>
          <label style={{ fontSize: '12px', color: '#666', fontWeight: 500, whiteSpace: 'nowrap' }}>
            Diff:
          </label>
          <select
            value={diffLensModel}
            onChange={(e) => setDiffLensModel(e.target.value)}
            className="font-mono"
            style={{
              padding: '6px 8px',
              border: '1px solid #ddd',
              borderRadius: '4px',
              fontSize: '12px',
              background: '#fff',
              cursor: 'pointer',
              minWidth: '180px',
            }}
          >
            <option value="">Select comparison model...</option>
            {AVAILABLE_MODELS.filter((m) => m !== conversation.model).map((model) => (
              <option key={model} value={model}>
                {model.split('/').pop()}
              </option>
            ))}
          </select>
          <button
            onClick={handleApplyDiff}
            disabled={!diffLensModel || isApplyingDiff || isGenerating}
            style={{
              padding: '6px 12px',
              borderRadius: '4px',
              border: '1px solid #ddd',
              background: diffLensModel && !isApplyingDiff ? '#fff' : '#f5f5f5',
              cursor: diffLensModel && !isApplyingDiff ? 'pointer' : 'default',
              fontSize: '12px',
              fontWeight: 500,
              color: diffLensModel && !isApplyingDiff ? '#333' : '#999',
            }}
          >
            {isApplyingDiff ? 'Applying...' : 'Apply Diff'}
          </button>
        </div>
      )}

      {/* Main input box container */}
      <div style={{
        border: '1px solid #ddd',
        borderRadius: '12px',
        background: '#fff',
        overflow: 'hidden',
      }}>
        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          disabled={isGenerating}
          style={{
            width: '100%',
            padding: '12px 16px',
            border: 'none',
            fontSize: '14px',
            lineHeight: '1.5',
            resize: 'none',
            minHeight: '44px',
            maxHeight: '150px',
            fontFamily: 'inherit',
            outline: 'none',
          }}
          rows={1}
        />

        {/* Bottom toolbar inside the box */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 12px',
          borderTop: '1px solid #eee',
          background: '#fafafa',
        }}>
          {/* Left side: Model + Temp */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {/* Model selector */}
            <select
              value={conversation.model}
              onChange={(e) => setModel(e.target.value)}
              className="font-mono"
              style={{
                padding: '4px 8px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontSize: '11px',
                background: '#fff',
                cursor: 'pointer',
                color: '#666',
              }}
            >
              {AVAILABLE_MODELS.map((model) => (
                <option key={model} value={model}>
                  {model.split('/').pop()}
                </option>
              ))}
            </select>

            {/* Temperature */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <span style={{ fontSize: '11px', color: '#999' }}>temp</span>
              <input
                type="number"
                value={conversation.temperature ?? 1.0}
                onChange={(e) => setTemperature(parseFloat(e.target.value) || 1.0)}
                min="0"
                max="2"
                step="0.1"
                style={{
                  width: '50px',
                  padding: '4px 6px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontSize: '11px',
                  textAlign: 'center',
                  background: '#fff',
                }}
              />
            </div>
          </div>

          {/* Right side: Send/Stop button */}
          {isGenerating ? (
            <button
              onClick={stopGeneration}
              style={{
                padding: '6px 16px',
                borderRadius: '6px',
                background: '#dc2626',
                border: 'none',
                cursor: 'pointer',
                color: '#fff',
                fontSize: '12px',
                fontWeight: 500,
              }}
            >
              ■ Stop
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!inputText.trim()}
              style={{
                padding: '6px 16px',
                borderRadius: '6px',
                background: inputText.trim() ? '#000' : '#e5e5e5',
                border: 'none',
                cursor: inputText.trim() ? 'pointer' : 'default',
                color: inputText.trim() ? '#fff' : '#999',
                fontSize: '12px',
                fontWeight: 500,
              }}
            >
              Send ↵
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
