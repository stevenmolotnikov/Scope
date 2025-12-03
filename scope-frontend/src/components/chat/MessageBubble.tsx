'use client';

import { useState, useCallback } from 'react';
import { TokenizedMessage } from './TokenizedMessage';
import { useConversationStore } from '@/stores/conversationStore';
import { useStreamingGeneration } from '@/hooks/useStreamingGeneration';
import type { Message } from '@/types';

interface MessageBubbleProps {
  message: Message;
  siblingCount: number;
  siblingIndex: number;
}

export function MessageBubble({ message, siblingCount, siblingIndex }: MessageBubbleProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editText, setEditText] = useState(message.content);
  const [showControls, setShowControls] = useState(false);

  const { navigateToSibling, updateMessageContent, isGenerating } = useConversationStore();
  const { regenerateFrom } = useStreamingGeneration();

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

  const handleSaveEdit = useCallback(() => {
    updateMessageContent(message.id, editText);
    setIsEditing(false);
  }, [message.id, editText, updateMessageContent]);

  const handleCancelEdit = useCallback(() => {
    setEditText(message.content);
    setIsEditing(false);
  }, [message.content]);

  return (
    <div 
      style={{
        marginBottom: '24px',
        animation: 'fadeIn 0.25s ease-out',
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
      }}
      onMouseEnter={() => setShowControls(true)}
      onMouseLeave={() => setShowControls(false)}
    >
      <div style={{ maxWidth: '85%', position: 'relative' }}>
        {/* Message content */}
        {isEditing ? (
          <div style={{
            background: 'rgba(255,255,255,0.9)',
            backdropFilter: 'blur(8px)',
            border: '1px solid rgba(0,0,0,0.2)',
            padding: '12px',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
          }}>
            <textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              style={{
                width: '100%',
                minHeight: '60px',
                padding: '10px',
                border: '1px solid rgba(0,0,0,0.1)',
                borderRadius: '6px',
                fontFamily: 'Inter, sans-serif',
                fontSize: '14px',
                lineHeight: '1.5',
                resize: 'vertical',
                marginBottom: '12px',
                outline: 'none',
              }}
              autoFocus
            />
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
              <button
                onClick={handleCancelEdit}
                style={{
                  padding: '6px 14px',
                  borderRadius: '6px',
                  border: '1px solid rgba(0,0,0,0.15)',
                  background: 'transparent',
                  fontSize: '13px',
                  fontWeight: 500,
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleSaveEdit}
                style={{
                  padding: '6px 14px',
                  borderRadius: '6px',
                  border: '1px solid #000',
                  background: '#000',
                  color: '#fff',
                  fontSize: '13px',
                  fontWeight: 500,
                  cursor: 'pointer',
                }}
              >
                Save
              </button>
            </div>
          </div>
        ) : isUser ? (
          <div style={{
            background: 'rgba(255,255,255,0.85)',
            backdropFilter: 'blur(8px)',
            color: '#000',
            border: '1px solid rgba(0,0,0,0.15)',
            padding: '12px 16px',
            borderRadius: '18px 18px 4px 18px',
            whiteSpace: 'pre-wrap',
            wordWrap: 'break-word',
            fontSize: '14px',
            lineHeight: '1.6',
            boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
            fontFamily: 'Inter, sans-serif',
          }}>
            {message.content}
          </div>
        ) : isAssistant ? (
          <div style={{
            background: 'rgba(250,250,250,0.8)',
            backdropFilter: 'blur(8px)',
            padding: '16px',
            borderRadius: '18px 18px 18px 4px',
            border: '1px solid rgba(0,0,0,0.08)',
            fontSize: '14px',
            lineHeight: '1.8',
            whiteSpace: 'pre-wrap',
            wordWrap: 'break-word',
            boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
          }}>
            {message.tokens && message.tokens.length > 0 ? (
              <TokenizedMessage tokens={message.tokens} messageId={message.id} />
            ) : (
              <span style={{ color: '#999' }}>{message.content || 'Generating...'}</span>
            )}
          </div>
        ) : null}

        {/* Message controls */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            marginTop: '4px',
            height: '24px',
            opacity: showControls ? 1 : 0,
            transition: 'opacity 0.15s ease',
            justifyContent: isUser ? 'flex-end' : 'flex-start',
          }}
        >
          {/* Sibling navigation */}
          {hasMultipleSiblings && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '2px',
              fontSize: '11px',
              color: '#666',
              background: '#fff',
              border: '1px solid rgba(0,0,0,0.1)',
              borderRadius: '4px',
              padding: '2px 4px',
              fontFamily: "'IBM Plex Mono', monospace",
            }}>
              <button
                onClick={() => handleNavigate('prev')}
                disabled={siblingIndex === 0}
                style={{
                  padding: '2px 6px',
                  background: 'transparent',
                  border: 'none',
                  cursor: siblingIndex === 0 ? 'default' : 'pointer',
                  opacity: siblingIndex === 0 ? 0.3 : 1,
                  borderRadius: '2px',
                }}
              >
                ‹
              </button>
              <span style={{ minWidth: '32px', textAlign: 'center' }}>
                {siblingIndex + 1}/{siblingCount}
              </span>
              <button
                onClick={() => handleNavigate('next')}
                disabled={siblingIndex === siblingCount - 1}
                style={{
                  padding: '2px 6px',
                  background: 'transparent',
                  border: 'none',
                  cursor: siblingIndex === siblingCount - 1 ? 'default' : 'pointer',
                  opacity: siblingIndex === siblingCount - 1 ? 0.3 : 1,
                  borderRadius: '2px',
                }}
              >
                ›
              </button>
            </div>
          )}

          {/* Edit button (user messages) */}
          {isUser && (
            <button
              onClick={() => setIsEditing(true)}
              style={{
                padding: '4px',
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                color: '#666',
                borderRadius: '4px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
              title="Edit message"
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path
                  d="M10 2L12 4L5 11H3V9L10 2Z"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          )}

          {/* Regenerate button (assistant messages) */}
          {isAssistant && (
            <button
              onClick={handleRegenerate}
              disabled={isGenerating}
              style={{
                padding: '4px',
                background: 'transparent',
                border: 'none',
                cursor: isGenerating ? 'default' : 'pointer',
                color: '#666',
                borderRadius: '4px',
                opacity: isGenerating ? 0.5 : 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
              title="Regenerate"
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path
                  d="M1 7C1 10.3137 3.68629 13 7 13C10.3137 13 13 10.3137 13 7C13 3.68629 10.3137 1 7 1C4.5 1 2.5 2.5 1.5 4.5M1 1V5H5"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
