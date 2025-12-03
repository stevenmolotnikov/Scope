import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Token, ViewMode, HighlightMode, SelectedTokenInfo } from '@/types';

interface ModalState {
  systemPrompt: boolean;
  sampling: boolean;
  rules: boolean;
  logitLens: boolean;
}

// Serializable token reference for persistence
interface PersistedTokenRef {
  messageId: string;
  tokenIndex: number;
}

interface UIState {
  // Hydration state
  _hasHydrated: boolean;
  setHasHydrated: (state: boolean) => void;

  // Sidebar state
  leftSidebarCollapsed: boolean;
  rightSidebarCollapsed: boolean;

  // View settings
  viewMode: ViewMode;
  highlightMode: HighlightMode;

  // Token selection
  selectedToken: SelectedTokenInfo | null;
  hoveredToken: SelectedTokenInfo | null;
  // Persisted reference to rehydrate selectedToken
  persistedTokenRef: PersistedTokenRef | null;

  // Modal state
  modals: ModalState;

  // Logit Lens state
  logitLensTokens: Token[];
  logitLensMessageId: string | null;
  logitLensTokenIndex: number;

  // DiffLens state
  diffLensEnabled: boolean;
  diffLensModel: string;

  // Prefill state
  prefillEnabled: boolean;
  prefillText: string;

  // Actions
  toggleLeftSidebar: () => void;
  toggleRightSidebar: () => void;
  setLeftSidebarCollapsed: (collapsed: boolean) => void;
  setRightSidebarCollapsed: (collapsed: boolean) => void;

  setViewMode: (mode: ViewMode) => void;
  setHighlightMode: (mode: HighlightMode) => void;

  selectToken: (info: SelectedTokenInfo | null) => void;
  setHoveredToken: (info: SelectedTokenInfo | null) => void;

  openModal: (modal: keyof ModalState) => void;
  closeModal: (modal: keyof ModalState) => void;
  closeAllModals: () => void;

  setLogitLensContext: (
    tokens: Token[],
    messageId: string,
    tokenIndex: number
  ) => void;
  clearLogitLensContext: () => void;

  setDiffLensEnabled: (enabled: boolean) => void;
  setDiffLensModel: (model: string) => void;

  setPrefillEnabled: (enabled: boolean) => void;
  setPrefillText: (text: string) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      // Hydration state
      _hasHydrated: false,
      setHasHydrated: (state: boolean) => set({ _hasHydrated: state }),

      // Initial state
      leftSidebarCollapsed: false,
      rightSidebarCollapsed: true,
      viewMode: 'token',
      highlightMode: 'probability',
      selectedToken: null,
      hoveredToken: null,
      persistedTokenRef: null,
      modals: {
        systemPrompt: false,
        sampling: false,
        rules: false,
        logitLens: false,
      },
      logitLensTokens: [],
      logitLensMessageId: null,
      logitLensTokenIndex: 0,
      diffLensEnabled: false,
      diffLensModel: '',
      prefillEnabled: false,
      prefillText: '',

      // Sidebar actions
      toggleLeftSidebar: () =>
        set((state) => ({ leftSidebarCollapsed: !state.leftSidebarCollapsed })),

      toggleRightSidebar: () =>
        set((state) => ({ rightSidebarCollapsed: !state.rightSidebarCollapsed })),

      setLeftSidebarCollapsed: (collapsed: boolean) =>
        set({ leftSidebarCollapsed: collapsed }),

      setRightSidebarCollapsed: (collapsed: boolean) =>
        set({ rightSidebarCollapsed: collapsed }),

      // View settings
      setViewMode: (mode: ViewMode) => set({ viewMode: mode }),

      setHighlightMode: (mode: HighlightMode) => set({ highlightMode: mode }),

      // Token selection - store reference for persistence
      selectToken: (info: SelectedTokenInfo | null) =>
        set({
          selectedToken: info,
          persistedTokenRef: info ? { messageId: info.messageId, tokenIndex: info.tokenIndex } : null,
          // Open sidebar when selecting, but don't force close when deselecting
          ...(info ? { rightSidebarCollapsed: false } : {}),
        }),

      setHoveredToken: (info: SelectedTokenInfo | null) =>
        set({ hoveredToken: info }),

      // Modal actions
      openModal: (modal: keyof ModalState) =>
        set((state) => ({
          modals: { ...state.modals, [modal]: true },
        })),

      closeModal: (modal: keyof ModalState) =>
        set((state) => ({
          modals: { ...state.modals, [modal]: false },
        })),

      closeAllModals: () =>
        set({
          modals: {
            systemPrompt: false,
            sampling: false,
            rules: false,
            logitLens: false,
          },
        }),

      // Logit Lens
      setLogitLensContext: (tokens, messageId, tokenIndex) =>
        set({
          logitLensTokens: tokens,
          logitLensMessageId: messageId,
          logitLensTokenIndex: tokenIndex,
          modals: {
            systemPrompt: false,
            sampling: false,
            rules: false,
            logitLens: true,
          },
        }),

      clearLogitLensContext: () =>
        set({
          logitLensTokens: [],
          logitLensMessageId: null,
          logitLensTokenIndex: 0,
        }),

      // DiffLens
      setDiffLensEnabled: (enabled: boolean) => set({ diffLensEnabled: enabled }),

      setDiffLensModel: (model: string) => set({ diffLensModel: model }),

      // Prefill
      setPrefillEnabled: (enabled: boolean) => set({ prefillEnabled: enabled }),

      setPrefillText: (text: string) => set({ prefillText: text }),
    }),
    {
      name: 'scope-ui',
      partialize: (state) => ({
        leftSidebarCollapsed: state.leftSidebarCollapsed,
        rightSidebarCollapsed: state.rightSidebarCollapsed,
        viewMode: state.viewMode,
        highlightMode: state.highlightMode,
        diffLensEnabled: state.diffLensEnabled,
        diffLensModel: state.diffLensModel,
        prefillEnabled: state.prefillEnabled,
        persistedTokenRef: state.persistedTokenRef,
      }),
      onRehydrateStorage: () => (state) => {
        state?.setHasHydrated(true);
      },
    }
  )
);
