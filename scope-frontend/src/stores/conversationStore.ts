import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  Conversation,
  ConversationsMap,
  Message,
  MessageTree,
  Token,
  GenerationRule,
  SamplingSettings,
} from '@/types';
import { uuidv4, generateConversationTitle } from '@/lib/utils';
import { api } from '@/lib/api';

const DEFAULT_MODEL = 'google/gemma-3-1b-it';
const DEFAULT_TEMPERATURE = 1.0;

interface ConversationState {
  // Data
  conversations: ConversationsMap;
  currentConversationId: string | null;
  isLoading: boolean;
  error: string | null;

  // Generation state
  isGenerating: boolean;
  abortController: AbortController | null;

  // Settings
  generationRules: GenerationRule[];
  samplingSettings: SamplingSettings;
  defaultSystemPrompt: string;

  // Actions
  loadConversations: () => Promise<void>;
  createNewChat: () => string;
  selectConversation: (id: string) => void;
  deleteConversation: (id: string) => Promise<void>;
  saveConversation: (conversation: Conversation) => Promise<void>;

  // Message tree operations
  addUserMessage: (content: string) => string;
  addAssistantMessage: (parentId: string) => string;
  updateMessageTokens: (messageId: string, tokens: Token[]) => void;
  updateMessageContent: (messageId: string, content: string) => void;
  navigateToSibling: (messageId: string, direction: 'prev' | 'next') => void;
  setCurrentLeaf: (messageId: string) => void;

  // Model/settings
  setModel: (model: string) => void;
  setTemperature: (temp: number) => void;
  setSystemPrompt: (prompt: string) => void;
  setGenerationRules: (rules: GenerationRule[]) => void;
  setSamplingSettings: (settings: SamplingSettings) => void;
  setDefaultSystemPrompt: (prompt: string) => void;

  // Generation control
  setGenerating: (generating: boolean, controller?: AbortController | null) => void;
  stopGeneration: () => void;

  // Helpers
  getCurrentConversation: () => Conversation | null;
  getConversationHistory: () => Array<{ role: string; content: string }>;
  getMessagePath: () => Message[];
}

export const useConversationStore = create<ConversationState>()(
  persist(
    (set, get) => ({
      // Initial state
      conversations: {},
      currentConversationId: null,
      isLoading: false,
      error: null,
      isGenerating: false,
      abortController: null,
      generationRules: [],
      samplingSettings: { top_k: 0, top_p: 1.0 },
      defaultSystemPrompt: '',

      // Load conversations from server
      loadConversations: async () => {
        set({ isLoading: true, error: null });
        try {
          const convs = await api.getConversations();
          set({ conversations: convs, isLoading: false });
        } catch (err) {
          set({ error: String(err), isLoading: false });
        }
      },

      // Create new chat
      createNewChat: () => {
        const id = `conv_${Date.now()}`;
        const rootId = uuidv4();
        const { defaultSystemPrompt } = get();

        const newConversation: Conversation = {
          id,
          title: 'New Chat',
          model: DEFAULT_MODEL,
          temperature: DEFAULT_TEMPERATURE,
          systemPrompt: defaultSystemPrompt,
          messageTree: {
            [rootId]: {
              id: rootId,
              role: 'system',
              content: '',
              parentId: null,
              childrenIds: [],
              activeChildIndex: 0,
            },
          },
          currentLeafId: rootId,
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };

        set((state) => ({
          conversations: { ...state.conversations, [id]: newConversation },
          currentConversationId: id,
        }));

        return id;
      },

      // Select conversation
      selectConversation: (id: string) => {
        set({ currentConversationId: id });
      },

      // Delete conversation
      deleteConversation: async (id: string) => {
        try {
          await api.deleteConversation(id);
          set((state) => {
            const { [id]: _, ...rest } = state.conversations;
            return {
              conversations: rest,
              currentConversationId:
                state.currentConversationId === id ? null : state.currentConversationId,
            };
          });
        } catch (err) {
          set({ error: String(err) });
        }
      },

      // Save conversation
      saveConversation: async (conversation: Conversation) => {
        try {
          await api.saveConversation(conversation.id, conversation);
          set((state) => ({
            conversations: {
              ...state.conversations,
              [conversation.id]: { ...conversation, updatedAt: Date.now() },
            },
          }));
        } catch (err) {
          set({ error: String(err) });
        }
      },

      // Add user message
      addUserMessage: (content: string) => {
        const { currentConversationId, conversations } = get();
        if (!currentConversationId) return '';

        const conv = conversations[currentConversationId];
        if (!conv) return '';

        const messageId = uuidv4();
        const parentId = conv.currentLeafId;

        const newMessage: Message = {
          id: messageId,
          role: 'user',
          content,
          parentId,
          childrenIds: [],
          activeChildIndex: 0,
          timestamp: Date.now(),
        };

        const updatedTree = { ...conv.messageTree };
        updatedTree[messageId] = newMessage;

        // Update parent's children
        if (parentId && updatedTree[parentId]) {
          updatedTree[parentId] = {
            ...updatedTree[parentId],
            childrenIds: [...updatedTree[parentId].childrenIds, messageId],
            activeChildIndex: updatedTree[parentId].childrenIds.length,
          };
        }

        // Update title if first user message
        const isFirstUserMessage = !Object.values(conv.messageTree).some(
          (m) => m.role === 'user'
        );
        const title = isFirstUserMessage ? generateConversationTitle(content) : conv.title;

        const updatedConv: Conversation = {
          ...conv,
          title,
          messageTree: updatedTree,
          currentLeafId: messageId,
          updatedAt: Date.now(),
        };

        set((state) => ({
          conversations: {
            ...state.conversations,
            [currentConversationId]: updatedConv,
          },
        }));

        // Auto-save
        api.saveConversation(currentConversationId, updatedConv);

        return messageId;
      },

      // Add assistant message placeholder
      addAssistantMessage: (parentId: string) => {
        const { currentConversationId, conversations } = get();
        if (!currentConversationId) return '';

        const conv = conversations[currentConversationId];
        if (!conv) return '';

        const messageId = uuidv4();

        const newMessage: Message = {
          id: messageId,
          role: 'assistant',
          content: '',
          tokens: [],
          parentId,
          childrenIds: [],
          activeChildIndex: 0,
          timestamp: Date.now(),
        };

        const updatedTree = { ...conv.messageTree };
        updatedTree[messageId] = newMessage;

        // Update parent's children
        if (updatedTree[parentId]) {
          updatedTree[parentId] = {
            ...updatedTree[parentId],
            childrenIds: [...updatedTree[parentId].childrenIds, messageId],
            activeChildIndex: updatedTree[parentId].childrenIds.length,
          };
        }

        const updatedConv: Conversation = {
          ...conv,
          messageTree: updatedTree,
          currentLeafId: messageId,
          updatedAt: Date.now(),
        };

        set((state) => ({
          conversations: {
            ...state.conversations,
            [currentConversationId]: updatedConv,
          },
        }));

        return messageId;
      },

      // Update message tokens during streaming
      updateMessageTokens: (messageId: string, tokens: Token[]) => {
        const { currentConversationId, conversations } = get();
        if (!currentConversationId) return;

        const conv = conversations[currentConversationId];
        if (!conv || !conv.messageTree[messageId]) return;

        const content = tokens.map((t) => t.token).join('');

        const updatedTree = {
          ...conv.messageTree,
          [messageId]: {
            ...conv.messageTree[messageId],
            tokens,
            content,
          },
        };

        set((state) => ({
          conversations: {
            ...state.conversations,
            [currentConversationId]: {
              ...conv,
              messageTree: updatedTree,
              updatedAt: Date.now(),
            },
          },
        }));
      },

      // Update message content (for editing)
      updateMessageContent: (messageId: string, content: string) => {
        const { currentConversationId, conversations } = get();
        if (!currentConversationId) return;

        const conv = conversations[currentConversationId];
        if (!conv || !conv.messageTree[messageId]) return;

        const updatedTree = {
          ...conv.messageTree,
          [messageId]: {
            ...conv.messageTree[messageId],
            content,
          },
        };

        const updatedConv: Conversation = {
          ...conv,
          messageTree: updatedTree,
          updatedAt: Date.now(),
        };

        set((state) => ({
          conversations: {
            ...state.conversations,
            [currentConversationId]: updatedConv,
          },
        }));

        api.saveConversation(currentConversationId, updatedConv);
      },

      // Navigate between message siblings
      navigateToSibling: (messageId: string, direction: 'prev' | 'next') => {
        const { currentConversationId, conversations } = get();
        if (!currentConversationId) return;

        const conv = conversations[currentConversationId];
        if (!conv) return;

        const message = conv.messageTree[messageId];
        if (!message || !message.parentId) return;

        const parent = conv.messageTree[message.parentId];
        if (!parent) return;

        const currentIndex = parent.childrenIds.indexOf(messageId);
        let newIndex = currentIndex;

        if (direction === 'prev' && currentIndex > 0) {
          newIndex = currentIndex - 1;
        } else if (direction === 'next' && currentIndex < parent.childrenIds.length - 1) {
          newIndex = currentIndex + 1;
        }

        if (newIndex === currentIndex) return;

        const newActiveChildId = parent.childrenIds[newIndex];

        // Find the leaf of this branch
        let leafId = newActiveChildId;
        let current = conv.messageTree[leafId];
        while (current && current.childrenIds.length > 0) {
          leafId = current.childrenIds[current.activeChildIndex];
          current = conv.messageTree[leafId];
        }

        const updatedTree = {
          ...conv.messageTree,
          [message.parentId]: {
            ...parent,
            activeChildIndex: newIndex,
          },
        };

        set((state) => ({
          conversations: {
            ...state.conversations,
            [currentConversationId]: {
              ...conv,
              messageTree: updatedTree,
              currentLeafId: leafId,
              updatedAt: Date.now(),
            },
          },
        }));
      },

      // Set current leaf
      setCurrentLeaf: (messageId: string) => {
        const { currentConversationId, conversations } = get();
        if (!currentConversationId) return;

        const conv = conversations[currentConversationId];
        if (!conv) return;

        set((state) => ({
          conversations: {
            ...state.conversations,
            [currentConversationId]: {
              ...conv,
              currentLeafId: messageId,
              updatedAt: Date.now(),
            },
          },
        }));
      },

      // Set model
      setModel: (model: string) => {
        const { currentConversationId, conversations } = get();
        if (!currentConversationId) return;

        const conv = conversations[currentConversationId];
        if (!conv) return;

        const updatedConv = { ...conv, model, updatedAt: Date.now() };

        set((state) => ({
          conversations: {
            ...state.conversations,
            [currentConversationId]: updatedConv,
          },
        }));

        api.saveConversation(currentConversationId, updatedConv);
      },

      // Set temperature
      setTemperature: (temperature: number) => {
        const { currentConversationId, conversations } = get();
        if (!currentConversationId) return;

        const conv = conversations[currentConversationId];
        if (!conv) return;

        const updatedConv = { ...conv, temperature, updatedAt: Date.now() };

        set((state) => ({
          conversations: {
            ...state.conversations,
            [currentConversationId]: updatedConv,
          },
        }));

        api.saveConversation(currentConversationId, updatedConv);
      },

      // Set system prompt
      setSystemPrompt: (systemPrompt: string) => {
        const { currentConversationId, conversations } = get();
        if (!currentConversationId) return;

        const conv = conversations[currentConversationId];
        if (!conv) return;

        const updatedConv = { ...conv, systemPrompt, updatedAt: Date.now() };

        set((state) => ({
          conversations: {
            ...state.conversations,
            [currentConversationId]: updatedConv,
          },
        }));

        api.saveConversation(currentConversationId, updatedConv);
      },

      // Set generation rules
      setGenerationRules: (rules: GenerationRule[]) => {
        set({ generationRules: rules });
      },

      // Set sampling settings
      setSamplingSettings: (settings: SamplingSettings) => {
        set({ samplingSettings: settings });
      },

      // Set default system prompt
      setDefaultSystemPrompt: (prompt: string) => {
        set({ defaultSystemPrompt: prompt });
      },

      // Set generating state
      setGenerating: (generating: boolean, controller?: AbortController | null) => {
        set({
          isGenerating: generating,
          abortController: controller ?? null,
        });
      },

      // Stop generation
      stopGeneration: () => {
        const { abortController } = get();
        if (abortController) {
          abortController.abort();
        }
        set({ isGenerating: false, abortController: null });
      },

      // Get current conversation
      getCurrentConversation: () => {
        const { currentConversationId, conversations } = get();
        if (!currentConversationId) return null;
        return conversations[currentConversationId] || null;
      },

      // Get conversation history for API
      getConversationHistory: () => {
        const conv = get().getCurrentConversation();
        if (!conv) return [];

        const messages: Array<{ role: string; content: string }> = [];

        // Add system prompt if exists
        if (conv.systemPrompt) {
          messages.push({ role: 'system', content: conv.systemPrompt });
        }

        // Build path from root to current leaf
        const path = get().getMessagePath();

        for (const msg of path) {
          if (msg.role !== 'system' && msg.content) {
            messages.push({ role: msg.role, content: msg.content });
          }
        }

        return messages;
      },

      // Get message path from root to current leaf
      getMessagePath: (): Message[] => {
        const conv = get().getCurrentConversation();
        if (!conv || !conv.currentLeafId) return [];

        const path: Message[] = [];
        let currentId: string | null = conv.currentLeafId;

        while (currentId) {
          const message: Message | undefined = conv.messageTree[currentId];
          if (!message) break;
          path.unshift(message);
          currentId = message.parentId;
        }

        return path;
      },
    }),
    {
      name: 'scope-conversations',
      partialize: (state) => ({
        defaultSystemPrompt: state.defaultSystemPrompt,
        generationRules: state.generationRules,
        samplingSettings: state.samplingSettings,
      }),
    }
  )
);

