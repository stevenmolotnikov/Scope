'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { TokenizedMessage } from './TokenizedMessage';
import { useConversationStore } from '@/stores/conversationStore';
import { useUIStore } from '@/stores/uiStore';
import { useStreamingGeneration } from '@/hooks/useStreamingGeneration';
import { Button } from '@/components/ui/Button';
import { X, ArrowUp, Pencil, RotateCcw } from 'lucide-react';
import type { Message } from '@/types';

const AVAILABLE_MODELS = [
  'google/gemma-3-1b-it',
  'google/gemma-2-2b-it',
];

interface MessageBubbleProps {
  message: Message;
  siblingCount: number;
  siblingIndex: number;
}

export function MessageBubble({ message, siblingCount, siblingIndex }: MessageBubbleProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState(message.content);
  const [showControls, setShowControls] = useState(false);
  const [editPrefillText, setEditPrefillText] = useState('');
  const editTextareaRef = useRef<HTMLTextAreaElement>(null);

  const { 
    navigateToSibling, 
    updateMessageContent, 
    isGenerating,
    getCurrentConversation,
    setModel,
    setTemperature,
  } = useConversationStore();
  
  const {
    prefillEnabled,
    diffLensEnabled,
    diffLensModel,
    setDiffLensModel,
  } = useUIStore();
  
  const { regenerateFrom } = useStreamingGeneration();
  
  const conversation = getCurrentConversation();

  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';
  const hasMultipleSiblings = siblingCount > 1;

  const handleNavigate = useCallback(
    (direction: 'prev' | 'next') => {
      navigateToSibling(message.id, direction);
    },
    [message.id, navigateToSibling]
  );

  const handleRegenerate = useCallback(() => {
    if (!message.parentId) return;
    regenerateFrom(message.parentId, {
      onError: (error) => console.error('Regeneration error:', error),
    });
  }, [message.parentId, regenerateFrom]);

  const handleCancelEdit = useCallback(() => {
    setEditText(message.content);
    setIsEditing(false);
    setEditPrefillText('');
  }, [message.content]);

  // Save and regenerate - updates message and triggers new generation
  const handleSubmitEdit = useCallback(() => {
    updateMessageContent(message.id, editText);
    setIsEditing(false);
    
    // Trigger regeneration from this user message
    regenerateFrom(message.id, {
      prefill: prefillEnabled && editPrefillText.trim() ? editPrefillText.trim() : undefined,
      diffModel: diffLensEnabled && diffLensModel ? diffLensModel : undefined,
      onError: (error) => console.error('Regeneration error:', error),
    });
    
    setEditPrefillText('');
  }, [message.id, editText, updateMessageContent, regenerateFrom, prefillEnabled, editPrefillText, diffLensEnabled, diffLensModel]);

  // Handle temperature change with formatting
  const handleTemperatureChange = (value: string) => {
    const floatVal = parseFloat(value);
    if (!isNaN(floatVal) && floatVal >= 0 && floatVal <= 2) {
      setTemperature(floatVal);
    }
  };

  const handleEditKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmitEdit();
      } else if (e.key === 'Escape') {
        handleCancelEdit();
      }
    },
    [handleSubmitEdit, handleCancelEdit]
  );

  // Auto-resize edit textarea
  useEffect(() => {
    if (editTextareaRef.current && isEditing) {
      editTextareaRef.current.style.height = 'auto';
      editTextareaRef.current.style.height = `${Math.min(editTextareaRef.current.scrollHeight, 200)}px`;
    }
  }, [editText, isEditing]);

  return (
    <div 
      className="mb-6 animate-fadeIn"
      onMouseEnter={() => setShowControls(true)}
      onMouseLeave={() => setShowControls(false)}
    >
      {/* Edit mode - Full width input matching ChatInput styling */}
      {isEditing && conversation ? (
        <div className="border border-border rounded-xl bg-background overflow-hidden shadow-sm focus-within:ring-1 focus-within:ring-ring focus-within:border-ring transition-all">
            <textarea
            ref={editTextareaRef}
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
            onKeyDown={handleEditKeyDown}
            className="w-full px-4 py-3 border-none text-sm resize-none min-h-[60px] max-h-[200px] bg-transparent focus:outline-none placeholder:text-muted-foreground"
              autoFocus
            />
          
          {/* Bottom toolbar */}
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

              {/* Right side: Actions */}
              <div className="flex items-center gap-2">
                <Button
                onClick={handleCancelEdit}
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-xs"
                >
                  <X className="h-3.5 w-3.5 mr-1" />
                Cancel
                </Button>
                <Button
                  onClick={handleSubmitEdit}
                  variant="primary"
                  size="sm"
                  className="h-7 px-3 text-xs"
                >
                  Submit
                  <ArrowUp className="h-3.5 w-3.5 ml-1" />
                </Button>
              </div>
            </div>

            {/* Advanced Controls (Prefill/Diff) */}
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
                      value={editPrefillText}
                      onChange={(e) => setEditPrefillText(e.target.value)}
                      placeholder="Start response with..."
                      className="font-mono flex-1 px-2 py-1 border border-input rounded-md text-[11px] bg-background focus:outline-none focus:ring-1 focus:ring-ring h-7"
                    />
                  </div>
                )}

                {/* Diff Section */}
                {diffLensEnabled && (
                  <div className="flex items-center gap-2">
                    <label className="w-12 text-[9px] font-semibold text-muted-foreground uppercase tracking-wider text-right">
                      Diff
                    </label>
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
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      ) : isEditing ? null : (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
          <div className="max-w-[85%] relative">
            {/* Message content */}
            {isUser ? (
              <div className="bg-background/85 backdrop-blur-sm text-foreground border border-border px-4 py-3 rounded-[18px_18px_4px_18px] whitespace-pre-wrap break-words text-sm leading-relaxed shadow-sm">
            {message.content}
          </div>
        ) : isAssistant ? (
              <div className="bg-muted/80 backdrop-blur-sm p-4 rounded-[18px_18px_18px_4px] border border-border/50 text-sm leading-7 whitespace-pre-wrap break-words shadow-sm">
            {message.tokens && message.tokens.length > 0 ? (
              <TokenizedMessage tokens={message.tokens} messageId={message.id} />
            ) : (
                  <span className="text-muted-foreground">{message.content || 'Generating...'}</span>
            )}
          </div>
        ) : null}

        {/* Message controls */}
        <div
              className={`flex items-center gap-2 mt-1 h-6 transition-opacity duration-150 ${
                showControls ? 'opacity-100' : 'opacity-0'
              } ${isUser ? 'justify-end' : 'justify-start'}`}
        >
          {/* Sibling navigation */}
          {hasMultipleSiblings && (
                <div className="flex items-center gap-0.5 text-[11px] text-muted-foreground bg-background border border-border rounded px-1 py-0.5 font-mono">
              <button
                onClick={() => handleNavigate('prev')}
                disabled={siblingIndex === 0}
                    className="px-1.5 py-0.5 bg-transparent border-none cursor-pointer disabled:cursor-default disabled:opacity-30 rounded-sm hover:bg-muted"
              >
                ‹
              </button>
                  <span className="min-w-[32px] text-center">
                {siblingIndex + 1}/{siblingCount}
              </span>
              <button
                onClick={() => handleNavigate('next')}
                disabled={siblingIndex === siblingCount - 1}
                    className="px-1.5 py-0.5 bg-transparent border-none cursor-pointer disabled:cursor-default disabled:opacity-30 rounded-sm hover:bg-muted"
              >
                ›
              </button>
            </div>
          )}

              {/* Edit button (user messages) */}
              {isUser && (
                <button
                  onClick={() => setIsEditing(true)}
                  className="p-1 bg-transparent border-none cursor-pointer text-muted-foreground rounded hover:bg-muted hover:text-foreground transition-colors"
                  title="Edit message"
                >
                  <Pencil size={14} />
                </button>
              )}

              {/* Regenerate button (assistant messages) */}
              {isAssistant && (
                <button
                  onClick={handleRegenerate}
                  disabled={isGenerating}
                  className="p-1 bg-transparent border-none cursor-pointer text-muted-foreground rounded hover:bg-muted hover:text-foreground transition-colors disabled:cursor-default disabled:opacity-50"
                  title="Regenerate"
                >
                  <RotateCcw size={14} />
                </button>
              )}
        </div>
      </div>
        </div>
      )}
    </div>
  );
}
