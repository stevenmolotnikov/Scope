'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { useConversationStore } from '@/stores/conversationStore';
import { useUIStore } from '@/stores/uiStore';
import { useStreamingGeneration } from '@/hooks/useStreamingGeneration';
import { api } from '@/lib/api';
import type { TokenDiffData, MessageRole } from '@/types';
import { Button } from '@/components/ui/Button';
import { cn } from '@/lib/utils';
import { ArrowUp, Square } from 'lucide-react';

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
            analysis_model: response.analysis_model,
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

  // Handle temperature change with formatting
  const handleTemperatureChange = (value: string) => {
    const floatVal = parseFloat(value);
    if (!isNaN(floatVal) && floatVal >= 0 && floatVal <= 2) {
      setTemperature(floatVal);
    }
  };

  if (!conversation) return null;

  return (
    <div className="border-t border-border bg-background p-6">
      {/* Main input box container */}
      <div className="border border-border rounded-xl bg-background overflow-hidden shadow-sm focus-within:ring-1 focus-within:ring-ring focus-within:border-ring transition-all">
        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          disabled={isGenerating}
          className="w-full px-4 py-3 border-none text-sm resize-none min-h-[44px] max-h-[150px] bg-transparent focus:outline-none placeholder:text-muted-foreground"
          rows={1}
        />

        {/* Bottom toolbar inside the box */}
        <div className="flex flex-col gap-3 px-3 py-2 border-t border-border bg-muted/30">
          <div className="flex items-center justify-between">
          {/* Left side: Model + Temp */}
            <div className="flex items-center gap-4">
            {/* Model selector */}
              <div className="flex flex-col gap-0.5">
                <label className="text-[9px] font-semibold text-muted-foreground uppercase tracking-wider pl-0.5">
                  Model
                </label>
            <select
              value={conversation.model}
              onChange={(e) => setModel(e.target.value)}
                  className="font-mono px-2 py-1 border border-input rounded-md text-[11px] bg-background focus:outline-none focus:ring-1 focus:ring-ring cursor-pointer text-foreground min-w-[140px]"
            >
              {AVAILABLE_MODELS.map((model) => (
                <option key={model} value={model}>
                  {model.split('/').pop()}
                </option>
              ))}
            </select>
              </div>

            {/* Temperature */}
              <div className="flex flex-col gap-0.5">
                <label className="text-[9px] font-semibold text-muted-foreground uppercase tracking-wider pl-0.5">
                  Temp
                </label>
              <input
                type="number"
                value={conversation.temperature ?? 1.0}
                  onChange={(e) => handleTemperatureChange(e.target.value)}
                  onBlur={(e) => {
                    const val = parseFloat(e.target.value);
                    if (!isNaN(val)) {
                      setTemperature(val); 
                      e.target.value = val.toFixed(1);
                    }
                  }}
                min="0"
                max="2"
                step="0.1"
                  className="w-14 px-1.5 py-1 border border-input rounded-md text-[11px] text-center bg-background focus:outline-none focus:ring-1 focus:ring-ring font-mono"
              />
            </div>
          </div>

          {/* Right side: Send/Stop button */}
          {isGenerating ? (
              <Button
              onClick={stopGeneration}
                variant="danger"
                size="sm"
                className="h-8 self-end"
              >
                <Square className="h-3 w-3 fill-current mr-1.5" />
                Stop
              </Button>
          ) : (
              <Button
              onClick={handleSubmit}
              disabled={!inputText.trim()}
                variant="primary"
                size="sm"
                className="h-8 self-end"
              >
                Send
                <ArrowUp className="h-3.5 w-3.5 ml-1.5" />
              </Button>
            )}
          </div>

          {/* Advanced Controls (Prefill/Diff) - Now below model/temp */}
          {(prefillEnabled || diffLensEnabled) && (
            <div className="flex flex-col gap-2 pt-2 border-t border-border/50">
              {/* Prefill Section */}
              {prefillEnabled && (
                <div className="flex items-center gap-2">
                  <label className="w-12 text-[9px] font-semibold text-muted-foreground uppercase tracking-wider text-right">
                    Prefill
                  </label>
                  <input
                    type="text"
                    value={prefillText}
                    onChange={(e) => setPrefillText(e.target.value)}
                    placeholder="Start response with..."
                    className="font-mono flex-1 px-2 py-1 border border-input rounded-md text-[11px] bg-background focus:outline-none focus:ring-1 focus:ring-ring h-7"
                  />
                </div>
              )}

              {/* Diff Section */}
              {diffLensEnabled && (
                <div className="flex items-center gap-2">
                  <label className="w-12 text-[9px] font-semibold text-muted-foreground uppercase tracking-wider text-right">
                    Diff Lens
                  </label>
                  <div className="flex items-center gap-2 flex-1">
                    <select
                      value={diffLensModel}
                      onChange={(e) => setDiffLensModel(e.target.value)}
                      className="font-mono flex-1 px-2 py-1 border border-input rounded-md text-[11px] bg-background focus:outline-none focus:ring-1 focus:ring-ring cursor-pointer h-7"
                    >
                      <option value="">Select comparison model...</option>
                      {AVAILABLE_MODELS.filter((m) => m !== conversation.model).map((model) => (
                        <option key={model} value={model}>
                          {model.split('/').pop()}
                        </option>
                      ))}
                    </select>
                    <Button
                      onClick={handleApplyDiff}
                      disabled={!diffLensModel || isApplyingDiff || isGenerating}
                      size="sm"
                      variant={diffLensModel && !isApplyingDiff ? "secondary" : "default"}
                      className="h-7 px-3 text-[10px] font-medium"
                    >
                      {isApplyingDiff ? 'Applying...' : 'Apply'}
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
