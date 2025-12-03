'use client';

import { Suspense } from 'react';
import { TopMenuBar } from './TopMenuBar';
import { ConversationsSidebar } from './sidebar/ConversationsSidebar';
import { TokenInspector } from './inspector/TokenInspector';
import { SystemPromptModal } from './modals/SystemPromptModal';
import { SamplingModal } from './modals/SamplingModal';
import { RulesModal } from './modals/RulesModal';
import { LogitLensModal } from './modals/LogitLensModal';
import { useUIStore } from '@/stores/uiStore';

interface AppLayoutProps {
  children: React.ReactNode;
}

export default function AppLayout({ children }: AppLayoutProps) {
  const { leftSidebarCollapsed } = useUIStore();

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      overflow: 'hidden',
    }}>
      {/* Top Menu Bar */}
      <TopMenuBar />

      {/* Main Content Area */}
      <div style={{
        display: 'flex',
        flex: 1,
        overflow: 'hidden',
      }}>
        {/* Left Sidebar */}
        <Suspense fallback={null}>
          <ConversationsSidebar />
        </Suspense>

        {/* Main Chat Area */}
        <main style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          marginLeft: leftSidebarCollapsed ? '40px' : 0,
          transition: 'margin-left 0.2s ease',
        }}>
          <Suspense fallback={
            <div style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#999',
            }}>
              Loading...
            </div>
          }>
            {children}
          </Suspense>
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

