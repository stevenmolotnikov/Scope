'use client';

import { useCallback, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useConversationStore } from '@/stores/conversationStore';
import { useUIStore } from '@/stores/uiStore';
import { Button } from '@/components/ui/Button';
import { cn, formatModelName } from '@/lib/utils';
import { 
  ChevronLeft, 
  ChevronRight, 
  Plus, 
  Trash2 
} from 'lucide-react';

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
      <div className="w-[48px] shrink-0 bg-background border-r border-border flex flex-col items-center pt-4">
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleLeftSidebar}
          className="h-8 w-8 text-muted-foreground hover:text-foreground"
          title="Expand sidebar"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <div className="w-[260px] shrink-0 flex flex-col bg-background border-r border-border h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-4 border-b border-border">
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleLeftSidebar}
          className="h-8 w-8 text-muted-foreground hover:text-foreground"
          title="Collapse sidebar"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
      </div>

      {/* New Chat Button */}
      <div className="p-4">
        <Button
          onClick={handleNewChat}
          className="w-full gap-2"
        >
          <Plus className="h-4 w-4" />
          New chat
        </Button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-auto px-3 pb-4">
        {sortedConversations.map((conv) => (
          <div
            key={conv.id}
            onClick={() => handleSelectConversation(conv.id)}
            className={cn(
              "group px-3 py-3 mb-1 rounded-md cursor-pointer flex items-center justify-between gap-2 text-sm transition-colors border",
              conv.id === currentConversationId
                ? "bg-accent/50 border-border font-medium text-foreground"
                : "bg-transparent border-transparent text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            )}
          >
            <div className="flex flex-col flex-1 min-w-0 gap-0.5">
              <span className="truncate">
                {conv.title || 'New Chat'}
              </span>
              <span className="text-[10px] text-muted-foreground/70 truncate font-mono">
                {formatModelName(conv.model)}
              </span>
            </div>
            <button
              onClick={(e) => handleDeleteConversation(conv.id, e)}
              className="p-1 text-muted-foreground hover:text-destructive opacity-0 group-hover:opacity-100 transition-opacity"
              title="Delete"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
