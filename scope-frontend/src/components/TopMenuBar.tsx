'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { useUIStore } from '@/stores/uiStore';
import { useConversationStore } from '@/stores/conversationStore';
import { useRouter } from 'next/navigation';

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
      label: 'Token View',
      onClick: () => setViewMode('token'),
      radio: true,
      checked: viewMode === 'token',
    },
    {
      label: 'Text View',
      onClick: () => setViewMode('text'),
      radio: true,
      checked: viewMode === 'text',
    },
    {
      label: 'Diff View',
      onClick: () => setViewMode('diff'),
      radio: true,
      checked: viewMode === 'diff',
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
      style={{
        height: '40px',
        display: 'flex',
        alignItems: 'center',
        padding: '0 12px',
        background: '#fff',
        borderBottom: '1px solid #e5e5e5',
        gap: '2px',
        position: 'relative',
        zIndex: 100,
        flexShrink: 0,
      }}
    >
      {/* Logo */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        marginRight: '16px',
        paddingRight: '16px',
        borderRight: '1px solid #e5e5e5',
      }}>
        <span style={{
          fontWeight: 700,
          fontSize: '14px',
          letterSpacing: '-0.01em',
          color: '#000',
        }}>
          Scope
        </span>
        <span style={{
          fontSize: '11px',
          color: '#999',
          fontWeight: 400,
        }}>
          by NDIF
        </span>
      </div>

      {/* Menu Buttons */}
      {menus.map((menu) => (
        <div key={menu.id} style={{ position: 'relative' }}>
          <button
            onClick={() => toggleMenu(menu.id)}
            onMouseEnter={() => openMenu && setOpenMenu(menu.id)}
            style={{
              padding: '4px 10px',
              background: openMenu === menu.id ? '#f0f0f0' : 'transparent',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: 400,
              color: '#333',
            }}
          >
            {menu.label}
          </button>

          {/* Dropdown */}
          {openMenu === menu.id && (
            <div
              style={{
                position: 'absolute',
                top: '100%',
                left: 0,
                marginTop: '2px',
                background: '#fff',
                border: '1px solid #e0e0e0',
                borderRadius: '6px',
                boxShadow: '0 4px 16px rgba(0,0,0,0.12), 0 0 0 1px rgba(0,0,0,0.04)',
                minWidth: '180px',
                padding: '4px 0',
                zIndex: 1000,
              }}
            >
              {menu.items.map((item, idx) =>
                item.divider ? (
                  <div
                    key={idx}
                    style={{
                      height: '1px',
                      background: '#eee',
                      margin: '4px 0',
                    }}
                  />
                ) : (
                  <button
                    key={idx}
                    onClick={() => handleMenuItemClick(item.onClick)}
                    disabled={item.disabled}
                    style={{
                      width: '100%',
                      padding: '6px 12px 6px 28px',
                      background: 'transparent',
                      border: 'none',
                      cursor: item.disabled ? 'default' : 'pointer',
                      fontSize: '13px',
                      textAlign: 'left',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      color: item.disabled ? '#999' : '#333',
                      position: 'relative',
                    }}
                    onMouseOver={(e) => {
                      if (!item.disabled) e.currentTarget.style.background = '#f5f5f5';
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.background = 'transparent';
                    }}
                  >
                    {/* Check/Radio indicator */}
                    {(item.checked !== undefined) && (
                      <span style={{
                        position: 'absolute',
                        left: '8px',
                        fontSize: '12px',
                        color: '#333',
                      }}>
                        {item.checked ? (item.radio ? '●' : '✓') : ''}
                      </span>
                    )}
                    <span>{item.label}</span>
                    {item.shortcut && (
                      <span style={{ 
                        color: '#999', 
                        fontSize: '11px',
                        marginLeft: '20px',
                      }}>
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
      <div style={{ flex: 1 }} />

      {/* Right side controls */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
      }}>
        {/* View mode pill */}
        <div style={{
          display: 'flex',
          background: '#f5f5f5',
          borderRadius: '6px',
          padding: '2px',
          gap: '2px',
        }}>
          <button
            onClick={() => setViewMode('token')}
            style={{
              padding: '4px 10px',
              fontSize: '12px',
              fontWeight: 500,
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              background: viewMode === 'token' ? '#fff' : 'transparent',
              color: viewMode === 'token' ? '#000' : '#666',
              boxShadow: viewMode === 'token' ? '0 1px 2px rgba(0,0,0,0.1)' : 'none',
            }}
          >
            Tokens
          </button>
          <button
            onClick={() => setViewMode('text')}
            style={{
              padding: '4px 10px',
              fontSize: '12px',
              fontWeight: 500,
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              background: viewMode === 'text' ? '#fff' : 'transparent',
              color: viewMode === 'text' ? '#000' : '#666',
              boxShadow: viewMode === 'text' ? '0 1px 2px rgba(0,0,0,0.1)' : 'none',
            }}
          >
            Text
          </button>
          <button
            onClick={() => setViewMode('diff')}
            style={{
              padding: '4px 10px',
              fontSize: '12px',
              fontWeight: 500,
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              background: viewMode === 'diff' ? '#fff' : 'transparent',
              color: viewMode === 'diff' ? '#000' : '#666',
              boxShadow: viewMode === 'diff' ? '0 1px 2px rgba(0,0,0,0.1)' : 'none',
            }}
          >
            Diff
          </button>
        </div>

        {/* Inspector toggle */}
        <button
          onClick={() => toggleRightSidebar()}
          title="Toggle Inspector"
          style={{
            width: '28px',
            height: '28px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: !rightSidebarCollapsed ? '#f0f0f0' : 'transparent',
            border: '1px solid #e5e5e5',
            borderRadius: '6px',
            cursor: 'pointer',
            color: '#666',
          }}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <rect x="2" y="3" width="12" height="10" rx="1.5" stroke="currentColor" strokeWidth="1.2" />
            <path d="M10 3V13" stroke="currentColor" strokeWidth="1.2" />
          </svg>
        </button>
      </div>
    </div>
  );
}
