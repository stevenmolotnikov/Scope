'use client';

import { useEffect, useRef } from 'react';
import { MessageBubble } from './MessageBubble';
import { useConversationStore } from '@/stores/conversationStore';
import type { Message } from '@/types';

export function MessagesContainer() {
  const containerRef = useRef<HTMLDivElement>(null);

  const { getCurrentConversation, getMessagePath, isGenerating } = useConversationStore();

  const conversation = getCurrentConversation();

  // Get the path from root to current leaf
  const messagePath: Message[] = conversation ? getMessagePath() : [];

  // Auto-scroll on new messages
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messagePath.length, isGenerating]);

  // Calculate sibling info for each message
  const getSiblingInfo = (message: Message, index: number) => {
    if (!conversation || !message.parentId) {
      return { siblingCount: 1, siblingIndex: 0 };
    }

    const parent = conversation.messageTree[message.parentId];
    if (!parent || !parent.childrenIds) {
      return { siblingCount: 1, siblingIndex: 0 };
    }

    return {
      siblingCount: parent.childrenIds.length,
      siblingIndex: parent.childrenIds.indexOf(message.id),
    };
  };

  if (!conversation) {
    return (
      <div style={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#999',
      }}>
        Select or create a conversation
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      style={{
        flex: 1,
        overflow: 'auto',
        padding: '24px 32px',
      }}
    >
      {/* Empty state */}
      {messagePath.length === 0 && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          color: '#999',
          textAlign: 'center',
        }}>
          <div style={{
            width: '64px',
            height: '64px',
            borderRadius: '50%',
            background: 'rgba(0,0,0,0.03)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: '16px',
          }}>
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <path
                d="M8 10C8 8.89543 8.89543 8 10 8H22C23.1046 8 24 8.89543 24 10V18C24 19.1046 23.1046 20 22 20H14L10 24V20H10C8.89543 20 8 19.1046 8 18V10Z"
                stroke="currentColor"
                strokeWidth="1.5"
              />
            </svg>
          </div>
          <p style={{ fontSize: '15px', fontWeight: 500 }}>Start a conversation</p>
          <p style={{ fontSize: '13px', marginTop: '4px' }}>
            Type a message below to begin
          </p>
        </div>
      )}

      {/* Messages */}
      <div style={{ maxWidth: '900px', margin: '0 auto' }}>
        {messagePath.map((message, index) => {
          const { siblingCount, siblingIndex } = getSiblingInfo(message, index);
          return (
            <MessageBubble
              key={message.id}
              message={message}
              siblingCount={siblingCount}
              siblingIndex={siblingIndex}
            />
          );
        })}

      </div>
    </div>
  );
}
