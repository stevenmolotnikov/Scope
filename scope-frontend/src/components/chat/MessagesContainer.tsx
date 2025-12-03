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
      <div className="flex-1 flex items-center justify-center text-muted-foreground">
        Select or create a conversation
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-auto px-8 py-6"
    >
      {/* Empty state */}
      {messagePath.length === 0 && (
        <div className="flex flex-col items-center justify-center h-full text-muted-foreground text-center">
          <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <path
                d="M8 10C8 8.89543 8.89543 8 10 8H22C23.1046 8 24 8.89543 24 10V18C24 19.1046 23.1046 20 22 20H14L10 24V20H10C8.89543 20 8 19.1046 8 18V10Z"
                stroke="currentColor"
                strokeWidth="1.5"
              />
            </svg>
          </div>
          <p className="text-[15px] font-medium text-foreground">Start a conversation</p>
          <p className="text-[13px] mt-1">
            Type a message below to begin
          </p>
        </div>
      )}

      {/* Messages */}
      <div className="max-w-[900px] mx-auto">
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
