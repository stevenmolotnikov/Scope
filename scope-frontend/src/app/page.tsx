'use client';

import { useEffect, useState, useRef } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { Suspense } from 'react';
import { TopMenuBar } from '@/components/TopMenuBar';
import { ConversationsSidebar } from '@/components/sidebar';
import { TokenInspector } from '@/components/inspector';
import { MessagesContainer, ChatInput } from '@/components/chat';
import {
  SystemPromptModal,
  SamplingModal,
  RulesModal,
  LogitLensModal,
} from '@/components/modals';
import { useConversationStore } from '@/stores/conversationStore';
import { useUIStore } from '@/stores/uiStore';

function HomeContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const conversationIdFromUrl = searchParams.get('conversationId');
  const [hasInitialized, setHasInitialized] = useState(false);
  const loadingRef = useRef(false);

  const {
    conversations,
    currentConversationId,
    loadConversations,
    selectConversation,
    createNewChat,
  } = useConversationStore();
  const { leftSidebarCollapsed } = useUIStore();

  // Load conversations on mount
  useEffect(() => {
    if (loadingRef.current) return;
    loadingRef.current = true;

    const init = async () => {
      await loadConversations();
      setHasInitialized(true);
    };
    init();
  }, [loadConversations]);

  // Handle URL parameter and select conversation AFTER loading completes
  useEffect(() => {
    if (!hasInitialized) return;

    const convIds = Object.keys(conversations);

    if (conversationIdFromUrl) {
      // URL has a conversation ID - try to select it
      if (conversations[conversationIdFromUrl]) {
        if (currentConversationId !== conversationIdFromUrl) {
          selectConversation(conversationIdFromUrl);
        }
      } else if (convIds.length > 0) {
        // Conversation not found, redirect to first available
        router.replace(`/?conversationId=${convIds[0]}`);
      } else {
        // No conversations exist, create new one
        const newId = createNewChat();
        router.replace(`/?conversationId=${newId}`);
      }
    } else {
      // No URL parameter
      if (convIds.length > 0) {
        // Select first conversation and update URL
        const firstId = convIds[0];
        if (currentConversationId !== firstId) {
          selectConversation(firstId);
        }
        router.replace(`/?conversationId=${firstId}`);
      } else {
        // No conversations, create new one
        const newId = createNewChat();
        router.replace(`/?conversationId=${newId}`);
      }
    }
  }, [
    hasInitialized,
    conversations,
    conversationIdFromUrl,
    currentConversationId,
    selectConversation,
    createNewChat,
    router,
  ]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      {/* Top Menu Bar */}
      <TopMenuBar />

      {/* Main Content */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left Sidebar - Conversations */}
        <ConversationsSidebar />

        {/* Chat Area */}
        <main style={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minWidth: 0,
          background: '#fff',
          overflow: 'hidden',
        }}>
          <MessagesContainer />
          <ChatInput />
        </main>

        {/* Right Sidebar - Token Inspector */}
        <TokenInspector />
      </div>

      {/* Modals */}
      <SystemPromptModal />
      <SamplingModal />
      <RulesModal />
      <LogitLensModal />
    </div>
  );
}

export default function HomePage() {
  return (
    <Suspense fallback={
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        color: '#999',
      }}>
        Loading...
      </div>
    }>
      <HomeContent />
    </Suspense>
  );
}
