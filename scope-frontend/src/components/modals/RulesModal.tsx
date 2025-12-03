'use client';

import { useState, useEffect, useCallback } from 'react';
import { Modal, Button } from '@/components/ui';
import { useUIStore } from '@/stores/uiStore';
import { useConversationStore } from '@/stores/conversationStore';
import { uuidv4 } from '@/lib/utils';
import type { GenerationRule } from '@/types';

const AVAILABLE_MODELS = [
  'google/gemma-3-1b-it',
  'Qwen/Qwen2.5-0.5B-Instruct',
  'meta-llama/Llama-3.2-1B-Instruct',
  'HuggingFaceTB/SmolLM2-360M-Instruct',
];

const DEFAULT_RULE: GenerationRule = {
  id: '',
  name: 'New Rule',
  enabled: true,
  criteria: {
    type: 'probability_below',
    threshold: 0.05,
  },
  action: {
    type: 'resample_same',
    strategy: 'sample',
    max_attempts: 3,
  },
};

export function RulesModal() {
  const { modals, closeModal } = useUIStore();
  const { generationRules, setGenerationRules } = useConversationStore();

  const [rules, setRules] = useState<GenerationRule[]>([]);

  useEffect(() => {
    if (modals.rules) {
      setRules(generationRules);
    }
  }, [modals.rules, generationRules]);

  const handleClose = () => {
    setGenerationRules(rules);
    closeModal('rules');
  };

  const handleAddRule = useCallback(() => {
    const newRule: GenerationRule = {
      ...DEFAULT_RULE,
      id: uuidv4(),
      name: `Rule ${rules.length + 1}`,
    };
    setRules([...rules, newRule]);
  }, [rules]);

  const handleDeleteRule = useCallback((id: string) => {
    setRules((prev) => prev.filter((r) => r.id !== id));
  }, []);

  const handleUpdateRule = useCallback(
    (id: string, updates: Partial<GenerationRule>) => {
      setRules((prev) =>
        prev.map((r) => (r.id === id ? { ...r, ...updates } : r))
      );
    },
    []
  );

  const inputStyle: React.CSSProperties = {
    width: '100%',
    padding: '8px 10px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    fontSize: '13px',
  };

  const selectStyle: React.CSSProperties = {
    ...inputStyle,
    background: '#fff',
    cursor: 'pointer',
  };

  const labelStyle: React.CSSProperties = {
    display: 'block',
    marginBottom: '4px',
    fontSize: '11px',
    fontWeight: 600,
    color: '#666',
    textTransform: 'uppercase',
  };

  return (
    <Modal
      isOpen={modals.rules}
      onClose={handleClose}
      title="Automation Rules"
      width="700px"
      footer={
        <>
          <Button variant="secondary" onClick={() => setRules([])}>
            Clear All
          </Button>
          <Button variant="primary" onClick={handleClose}>
            Done
          </Button>
        </>
      }
    >
      <div>
        {/* Header */}
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'flex-start',
          marginBottom: '20px',
        }}>
          <div>
            <p style={{ fontSize: '13px', color: '#666', margin: 0 }}>
              Create rules that watch tokens during generation and react when they match your criteria.
            </p>
          </div>
          <Button onClick={handleAddRule}>+ Add Rule</Button>
        </div>

        {/* Rules list */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {rules.length === 0 && (
            <div style={{
              textAlign: 'center',
              padding: '40px 20px',
              border: '2px dashed #e5e5e5',
              borderRadius: '8px',
              color: '#999',
            }}>
              <p style={{ marginBottom: '12px' }}>No rules configured</p>
              <Button onClick={handleAddRule}>Add your first rule</Button>
            </div>
          )}

          {rules.map((rule) => (
            <div
              key={rule.id}
              style={{
                border: '1px solid #e5e5e5',
                borderRadius: '8px',
                padding: '16px',
                background: rule.enabled ? '#fff' : '#fafafa',
                opacity: rule.enabled ? 1 : 0.7,
              }}
            >
              {/* Rule header */}
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '12px',
                marginBottom: '16px',
              }}>
                {/* Toggle */}
                <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={rule.enabled}
                    onChange={(e) => handleUpdateRule(rule.id, { enabled: e.target.checked })}
                    style={{ marginRight: '8px' }}
                  />
                </label>

                {/* Name */}
                <input
                  type="text"
                  value={rule.name}
                  onChange={(e) => handleUpdateRule(rule.id, { name: e.target.value })}
                  style={{
                    flex: 1,
                    padding: '8px 12px',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    fontSize: '14px',
                    fontWeight: 500,
                  }}
                />

                {/* Delete */}
                <button
                  onClick={() => handleDeleteRule(rule.id)}
                  style={{
                    padding: '6px 10px',
                    background: 'transparent',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    color: '#999',
                    fontSize: '12px',
                  }}
                >
                  Delete
                </button>
              </div>

              {/* When section */}
              <div style={{
                background: '#f9f9f9',
                border: '1px solid #eee',
                borderRadius: '6px',
                padding: '12px',
                marginBottom: '12px',
              }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#999', marginBottom: '10px', textTransform: 'uppercase' }}>
                  When
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div>
                    <label style={labelStyle}>Criteria Type</label>
                    <select
                      value={rule.criteria.type}
                      onChange={(e) => handleUpdateRule(rule.id, {
                        criteria: { ...rule.criteria, type: e.target.value as GenerationRule['criteria']['type'] }
                      })}
                      style={selectStyle}
                    >
                      <option value="probability_below">Probability Below</option>
                      <option value="consecutive_probability_below">Consecutive Low Prob</option>
                      <option value="text_match">Text Match</option>
                    </select>
                  </div>

                  {(rule.criteria.type === 'probability_below' || rule.criteria.type === 'consecutive_probability_below') && (
                    <div>
                      <label style={labelStyle}>Threshold</label>
                      <input
                        type="number"
                        value={rule.criteria.threshold ?? 0.05}
                        onChange={(e) => handleUpdateRule(rule.id, {
                          criteria: { ...rule.criteria, threshold: parseFloat(e.target.value) }
                        })}
                        min={0}
                        max={1}
                        step={0.01}
                        style={inputStyle}
                      />
                    </div>
                  )}

                  {rule.criteria.type === 'text_match' && (
                    <div>
                      <label style={labelStyle}>Match Value</label>
                      <input
                        type="text"
                        value={rule.criteria.value ?? ''}
                        onChange={(e) => handleUpdateRule(rule.id, {
                          criteria: { ...rule.criteria, value: e.target.value }
                        })}
                        placeholder="Text to match..."
                        style={inputStyle}
                      />
                    </div>
                  )}
                </div>
              </div>

              {/* Then section */}
              <div style={{
                background: '#f9f9f9',
                border: '1px solid #eee',
                borderRadius: '6px',
                padding: '12px',
              }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#999', marginBottom: '10px', textTransform: 'uppercase' }}>
                  Then
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div>
                    <label style={labelStyle}>Action</label>
                    <select
                      value={rule.action.type}
                      onChange={(e) => handleUpdateRule(rule.id, {
                        action: { ...rule.action, type: e.target.value as GenerationRule['action']['type'] }
                      })}
                      style={selectStyle}
                    >
                      <option value="resample_same">Resample (Same Model)</option>
                      <option value="resample_other_model">Resample (Other Model)</option>
                      <option value="replace_text">Replace with Text</option>
                    </select>
                  </div>

                  {rule.action.type === 'resample_other_model' && (
                    <div>
                      <label style={labelStyle}>Model</label>
                      <select
                        value={rule.action.model ?? ''}
                        onChange={(e) => handleUpdateRule(rule.id, {
                          action: { ...rule.action, model: e.target.value }
                        })}
                        style={selectStyle}
                      >
                        <option value="">Select model...</option>
                        {AVAILABLE_MODELS.map((m) => (
                          <option key={m} value={m}>{m.split('/').pop()}</option>
                        ))}
                      </select>
                    </div>
                  )}

                  {rule.action.type === 'replace_text' && (
                    <div>
                      <label style={labelStyle}>Replacement Text</label>
                      <input
                        type="text"
                        value={rule.action.text ?? ''}
                        onChange={(e) => handleUpdateRule(rule.id, {
                          action: { ...rule.action, text: e.target.value }
                        })}
                        placeholder="Text to insert..."
                        style={inputStyle}
                      />
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Modal>
  );
}
