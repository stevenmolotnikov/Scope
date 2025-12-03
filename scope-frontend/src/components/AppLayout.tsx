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
import { cn } from '@/lib/utils';

interface AppLayoutProps {
  children: React.ReactNode;
}

export default function AppLayout({ children }: AppLayoutProps) {
  const { leftSidebarCollapsed } = useUIStore();

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-background text-foreground">
      {/* Top Menu Bar */}
      <TopMenuBar />

      {/* Main Content Area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <Suspense fallback={null}>
          <ConversationsSidebar />
        </Suspense>

        {/* Main Chat Area */}
        <main 
          className={cn(
            "flex flex-1 flex-col overflow-hidden transition-[margin] duration-200 ease-in-out",
            leftSidebarCollapsed ? "ml-[40px]" : "ml-0"
          )}
        >
          <Suspense fallback={
            <div className="flex flex-1 items-center justify-center text-muted-foreground">
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
