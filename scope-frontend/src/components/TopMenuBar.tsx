'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { useUIStore } from '@/stores/uiStore';
import { useConversationStore } from '@/stores/conversationStore';
import { useRouter } from 'next/navigation';
import { cn } from '@/lib/utils';
import { Microscope } from 'lucide-react';

type MenuId = 'file' | 'view' | 'generation' | 'analysis' | null;

interface MenuItem {
  label: string;
  onClick: () => void;
  divider?: boolean;
  shortcut?: string;
  disabled?: boolean;
  checked?: boolean;
  radio?: boolean;
}

export function TopMenuBar() {
  const [openMenu, setOpenMenu] = useState<MenuId>(null);
  const menuBarRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  const {
    viewMode,
    highlightMode,
    setViewMode,
    setHighlightMode,
    leftSidebarCollapsed,
    rightSidebarCollapsed,
    toggleLeftSidebar,
    toggleRightSidebar,
    openModal,
    prefillEnabled,
    setPrefillEnabled,
    diffLensEnabled,
    setDiffLensEnabled,
    theme,
    setTheme,
  } = useUIStore();

  const { createNewChat, getCurrentConversation, saveConversation } = useConversationStore();

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuBarRef.current && !menuBarRef.current.contains(e.target as Node)) {
        setOpenMenu(null);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const toggleMenu = useCallback((menu: MenuId) => {
    setOpenMenu((prev) => (prev === menu ? null : menu));
  }, []);

  const handleMenuItemClick = useCallback((action: () => void) => {
    action();
    setOpenMenu(null);
  }, []);

  const fileMenuItems: MenuItem[] = [
    {
      label: 'New Chat',
      shortcut: '⌘N',
      onClick: () => {
        const newId = createNewChat();
        router.push(`/?conversationId=${newId}`);
      },
    },
    {
      label: 'Save',
      shortcut: '⌘S',
      onClick: () => {
        const conv = getCurrentConversation();
        if (conv) saveConversation(conv);
      },
    },
  ];

  const viewMenuItems: MenuItem[] = [
    {
      label: 'Conversations',
      onClick: toggleLeftSidebar,
      checked: !leftSidebarCollapsed,
    },
    {
      label: 'Inspector',
      onClick: toggleRightSidebar,
      checked: !rightSidebarCollapsed,
    },
    { label: '', onClick: () => {}, divider: true },
    {
      label: 'Perplexity Lens',
      onClick: () => setViewMode('token'),
      radio: true,
      checked: viewMode === 'token',
    },
    {
      label: 'Diff Lens',
      onClick: () => setViewMode('diff'),
      radio: true,
      checked: viewMode === 'diff',
    },
    {
      label: 'Text Lens',
      onClick: () => setViewMode('text'),
      radio: true,
      checked: viewMode === 'text',
    },
    { label: '', onClick: () => {}, divider: true },
    {
      label: 'Color by Probability',
      onClick: () => setHighlightMode('probability'),
      radio: true,
      checked: highlightMode === 'probability',
    },
    {
      label: 'Color by Rank',
      onClick: () => setHighlightMode('rank'),
      radio: true,
      checked: highlightMode === 'rank',
    },
    { label: '', onClick: () => {}, divider: true },
    {
      label: 'Light Mode',
      onClick: () => setTheme('light'),
      radio: true,
      checked: theme === 'light',
    },
    {
      label: 'Dark Mode',
      onClick: () => setTheme('dark'),
      radio: true,
      checked: theme === 'dark',
    },
    {
      label: 'System Theme',
      onClick: () => setTheme('system'),
      radio: true,
      checked: theme === 'system',
    },
  ];

  const generationMenuItems: MenuItem[] = [
    {
      label: 'System Prompt...',
      onClick: () => openModal('systemPrompt'),
    },
    {
      label: 'Sampling Settings...',
      onClick: () => openModal('sampling'),
    },
    {
      label: 'Automation Rules...',
      onClick: () => openModal('rules'),
    },
  ];

  const analysisMenuItems: MenuItem[] = [
    {
      label: 'Enable Prefill',
      onClick: () => setPrefillEnabled(!prefillEnabled),
      checked: prefillEnabled,
    },
    {
      label: 'Enable Diff',
      onClick: () => setDiffLensEnabled(!diffLensEnabled),
      checked: diffLensEnabled,
    },
  ];

  const menus: { id: MenuId; label: string; items: MenuItem[] }[] = [
    { id: 'file', label: 'File', items: fileMenuItems },
    { id: 'view', label: 'View', items: viewMenuItems },
    { id: 'generation', label: 'Generation', items: generationMenuItems },
    { id: 'analysis', label: 'Analysis', items: analysisMenuItems },
  ];

  return (
    <div
      ref={menuBarRef}
      className="h-10 flex items-center px-3 bg-background border-b border-border gap-0.5 relative z-10 shrink-0"
    >
      {/* Logo */}
      <div className="flex items-center gap-2 mr-4 pr-4 border-r border-border">
        <Microscope size={18} className="text-foreground" />
        <div className="flex items-baseline gap-1.5">
          <span className="font-semibold text-sm tracking-tight text-foreground">
          Scope
        </span>
          <span className="text-[10px] text-muted-foreground/70 font-normal">
          by NDIF
        </span>
        </div>
      </div>

      {/* Menu Buttons */}
      {menus.map((menu) => (
        <div key={menu.id} className="relative">
          <button
            onClick={() => toggleMenu(menu.id)}
            onMouseEnter={() => openMenu && setOpenMenu(menu.id)}
            className={cn(
              "px-2.5 py-1 text-[13px] font-normal rounded cursor-pointer transition-colors border-none",
              openMenu === menu.id
                ? "bg-accent text-accent-foreground"
                : "bg-transparent text-foreground hover:bg-accent hover:text-accent-foreground"
            )}
          >
            {menu.label}
          </button>

          {/* Dropdown */}
          {openMenu === menu.id && (
            <div className="absolute top-full left-0 mt-0.5 bg-popover border border-border rounded-md shadow-lg min-w-[180px] py-1 z-50">
              {menu.items.map((item, idx) =>
                item.divider ? (
                  <div
                    key={idx}
                    className="h-px bg-border my-1"
                  />
                ) : (
                  <button
                    key={idx}
                    onClick={() => handleMenuItemClick(item.onClick)}
                    disabled={item.disabled}
                    className={cn(
                      "w-full px-3 py-1.5 text-left text-[13px] flex justify-between items-center relative pl-7 border-none bg-transparent cursor-pointer transition-colors",
                      item.disabled
                        ? "opacity-50 pointer-events-none text-muted-foreground"
                        : "text-foreground hover:bg-accent hover:text-accent-foreground"
                    )}
                  >
                    {/* Check/Radio indicator */}
                    {(item.checked !== undefined) && (
                      <span className="absolute left-2 text-xs text-foreground">
                        {item.checked ? (item.radio ? '●' : '✓') : ''}
                      </span>
                    )}
                    <span>{item.label}</span>
                    {item.shortcut && (
                      <span className="text-muted-foreground text-[11px] ml-5">
                        {item.shortcut}
                      </span>
                    )}
                  </button>
                )
              )}
            </div>
          )}
        </div>
      ))}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Right side controls */}
      <div className="flex items-center gap-2">
        {/* View mode pill */}
        <div className="flex items-center bg-muted rounded-md p-0.5 pl-2 gap-1">
          <span className="text-[10px] font-medium text-muted-foreground/70 uppercase tracking-wide pr-1.5 border-r border-border/50 mr-0.5">
            Lenses
          </span>
          <button
            onClick={() => setViewMode('token')}
            className={cn(
              "px-2.5 py-1 text-xs font-medium border-none rounded-sm cursor-pointer transition-all",
              viewMode === 'token'
                ? "bg-background text-foreground shadow-sm"
                : "bg-transparent text-muted-foreground hover:text-foreground"
            )}
          >
            Perplexity
          </button>
          <button
            onClick={() => setViewMode('diff')}
            className={cn(
              "px-2.5 py-1 text-xs font-medium border-none rounded-sm cursor-pointer transition-all",
              viewMode === 'diff'
                ? "bg-background text-foreground shadow-sm"
                : "bg-transparent text-muted-foreground hover:text-foreground"
            )}
          >
            Diff
          </button>
          <button
            onClick={() => setViewMode('text')}
            className={cn(
              "px-2.5 py-1 text-xs font-medium border-none rounded-sm cursor-pointer transition-all",
              viewMode === 'text'
                ? "bg-background text-foreground shadow-sm"
                : "bg-transparent text-muted-foreground hover:text-foreground"
            )}
          >
            Text
          </button>
        </div>
      </div>
    </div>
  );
}
