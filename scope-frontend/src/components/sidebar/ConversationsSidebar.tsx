'use client';

import { useCallback, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useConversationStore } from '@/stores/conversationStore';
import { useUIStore } from '@/stores/uiStore';

export function ConversationsSidebar() {
  const router = useRouter();

  const {
    conversations,
    currentConversationId,
    loadConversations,
    createNewChat,
    selectConversation,
    deleteConversation,
  } = useConversationStore();
  const { leftSidebarCollapsed, toggleLeftSidebar } = useUIStore();

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  const handleNewChat = useCallback(() => {
    const newId = createNewChat();
    router.push(`/?conversationId=${newId}`);
  }, [createNewChat, router]);

  const handleSelectConversation = useCallback(
    (id: string) => {
      selectConversation(id);
      router.push(`/?conversationId=${id}`);
    },
    [selectConversation, router]
  );

  const handleDeleteConversation = useCallback(
    async (id: string, e: React.MouseEvent) => {
      e.stopPropagation();
      await deleteConversation(id);
      if (currentConversationId === id) {
        const remaining = Object.keys(conversations).filter((key) => key !== id);
        if (remaining.length > 0) {
          router.push(`/?conversationId=${remaining[0]}`);
        } else {
          handleNewChat();
        }
      }
    },
    [conversations, currentConversationId, deleteConversation, router, handleNewChat]
  );

  // Sort conversations by updatedAt (most recent first)
  const sortedConversations = Object.values(conversations).sort(
    (a, b) => (b.updatedAt || 0) - (a.updatedAt || 0)
  );

  // Collapsed state - just show toggle button
  if (leftSidebarCollapsed) {
    return (
      <div style={{
        width: '48px',
        flexShrink: 0,
        background: '#fff',
        borderRight: '1px solid #e5e5e5',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        paddingTop: '16px',
      }}>
        <button
          onClick={toggleLeftSidebar}
          style={{
            width: '32px',
            height: '32px',
            border: '1px solid #e5e5e5',
            borderRadius: '6px',
            background: '#fff',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#666',
          }}
          title="Expand sidebar"
        >
          ▶
        </button>
      </div>
    );
  }

  return (
    <div style={{
      width: '260px',
      flexShrink: 0,
      display: 'flex',
      flexDirection: 'column',
      background: '#fff',
      borderRight: '1px solid #e5e5e5',
      height: '100%',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '16px 16px 12px 16px',
        borderBottom: '1px solid #e5e5e5',
      }}>
        <button
          onClick={toggleLeftSidebar}
          style={{
            width: '28px',
            height: '28px',
            border: 'none',
            background: 'transparent',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#666',
            borderRadius: '4px',
          }}
          title="Collapse sidebar"
        >
          ◀
        </button>
      </div>

      {/* New Chat Button */}
      <div style={{ padding: '16px' }}>
        <button
          onClick={handleNewChat}
          style={{
            width: '100%',
            padding: '10px 16px',
            background: '#000',
            color: '#fff',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: 500,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
          }}
        >
          <span style={{ fontSize: '16px' }}>+</span>
          New chat
        </button>
      </div>

      {/* Conversations List */}
      <div style={{
        flex: 1,
        overflow: 'auto',
        padding: '0 12px 16px 12px',
      }}>
        {sortedConversations.map((conv) => (
          <div
            key={conv.id}
            onClick={() => handleSelectConversation(conv.id)}
            style={{
              padding: '12px 12px',
              marginBottom: '4px',
              borderRadius: '8px',
              cursor: 'pointer',
              background: conv.id === currentConversationId ? '#f0f0f0' : 'transparent',
              border: conv.id === currentConversationId ? '1px solid #e0e0e0' : '1px solid transparent',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '8px',
              fontSize: '14px',
              transition: 'background 0.1s',
            }}
            onMouseOver={(e) => {
              if (conv.id !== currentConversationId) {
                e.currentTarget.style.background = '#f5f5f5';
              }
            }}
            onMouseOut={(e) => {
              if (conv.id !== currentConversationId) {
                e.currentTarget.style.background = 'transparent';
              }
            }}
          >
            <span style={{
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              flex: 1,
              fontWeight: conv.id === currentConversationId ? 500 : 400,
              color: '#333',
            }}>
              {conv.title || 'New Chat'}
            </span>
            <button
              onClick={(e) => handleDeleteConversation(conv.id, e)}
              style={{
                padding: '4px 6px',
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                color: '#999',
                borderRadius: '4px',
                opacity: 0.5,
                fontSize: '12px',
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.opacity = '1';
                e.currentTarget.style.color = '#c00';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.opacity = '0.5';
                e.currentTarget.style.color = '#999';
              }}
              title="Delete"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
