'use client';

import { useState, useEffect } from 'react';
import { Modal, Button } from '@/components/ui';
import { useUIStore } from '@/stores/uiStore';
import { useConversationStore } from '@/stores/conversationStore';

export function SystemPromptModal() {
  const { modals, closeModal } = useUIStore();
  const {
    getCurrentConversation,
    setSystemPrompt,
    defaultSystemPrompt,
    setDefaultSystemPrompt,
  } = useConversationStore();

  const [promptText, setPromptText] = useState('');

  const conversation = getCurrentConversation();

  // Load current system prompt when modal opens
  useEffect(() => {
    if (modals.systemPrompt) {
      setPromptText(conversation?.systemPrompt || defaultSystemPrompt || '');
    }
  }, [modals.systemPrompt, conversation?.systemPrompt, defaultSystemPrompt]);

  const handleSave = () => {
    setSystemPrompt(promptText);
    closeModal('systemPrompt');
  };

  const handleSaveAsDefault = () => {
    setDefaultSystemPrompt(promptText);
    setSystemPrompt(promptText);
    closeModal('systemPrompt');
  };

  const handleClose = () => {
    closeModal('systemPrompt');
  };

  return (
    <Modal
      isOpen={modals.systemPrompt}
      onClose={handleClose}
      title="System Prompt"
      width="600px"
      footer={
        <>
          <Button variant="secondary" onClick={handleClose}>
            Cancel
          </Button>
          <Button variant="default" onClick={handleSave}>
            Save
          </Button>
          <Button variant="primary" onClick={handleSaveAsDefault}>
            Save as Default
          </Button>
        </>
      }
    >
      <div>
        <p style={{ marginBottom: '12px', fontSize: '13px', color: '#666' }}>
          The system prompt sets the behavior and context for the AI assistant.
        </p>
        <textarea
          value={promptText}
          onChange={(e) => setPromptText(e.target.value)}
          placeholder="e.g., You are a helpful assistant..."
          rows={10}
          style={{
            width: '100%',
            padding: '12px',
            border: '1px solid #ddd',
            borderRadius: '8px',
            fontSize: '14px',
            lineHeight: '1.5',
            resize: 'vertical',
            minHeight: '180px',
            fontFamily: 'inherit',
          }}
        />
      </div>
    </Modal>
  );
}
