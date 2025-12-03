'use client';

import { useState, useEffect } from 'react';
import { Modal, Button } from '@/components/ui';
import { useUIStore } from '@/stores/uiStore';
import { useConversationStore } from '@/stores/conversationStore';
import type { SamplingSettings } from '@/types';

export function SamplingModal() {
  const { modals, closeModal } = useUIStore();
  const { samplingSettings, setSamplingSettings } = useConversationStore();

  const [settings, setSettings] = useState<SamplingSettings>({
    top_k: 0,
    top_p: 1.0,
  });

  useEffect(() => {
    if (modals.sampling) {
      setSettings(samplingSettings);
    }
  }, [modals.sampling, samplingSettings]);

  const handleSave = () => {
    setSamplingSettings(settings);
    closeModal('sampling');
  };

  const handleClose = () => {
    closeModal('sampling');
  };

  const inputStyle: React.CSSProperties = {
    width: '100%',
    padding: '10px 12px',
    border: '1px solid #ddd',
    borderRadius: '6px',
    fontSize: '14px',
  };

  const labelStyle: React.CSSProperties = {
    display: 'block',
    marginBottom: '6px',
    fontSize: '13px',
    fontWeight: 500,
    color: '#333',
  };

  const helpStyle: React.CSSProperties = {
    fontSize: '12px',
    color: '#666',
    marginTop: '4px',
  };

  return (
    <Modal
      isOpen={modals.sampling}
      onClose={handleClose}
      title="Sampling Settings"
      width="400px"
      footer={
        <>
          <Button variant="secondary" onClick={handleClose}>
            Cancel
          </Button>
          <Button variant="primary" onClick={handleSave}>
            Save
          </Button>
        </>
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {/* Top-K */}
        <div>
          <label style={labelStyle}>Top-K</label>
          <input
            type="number"
            value={settings.top_k}
            onChange={(e) =>
              setSettings({ ...settings, top_k: parseInt(e.target.value) || 0 })
            }
            min="0"
            max="100"
            style={inputStyle}
          />
          <p style={helpStyle}>
            Limits sampling to top K tokens. Set to 0 to disable.
          </p>
        </div>

        {/* Top-P */}
        <div>
          <label style={labelStyle}>Top-P (Nucleus Sampling)</label>
          <input
            type="number"
            value={settings.top_p}
            onChange={(e) =>
              setSettings({ ...settings, top_p: parseFloat(e.target.value) || 1.0 })
            }
            min="0"
            max="1"
            step="0.05"
            style={inputStyle}
          />
          <p style={helpStyle}>
            Cumulative probability threshold. Set to 1.0 to disable.
          </p>
        </div>
      </div>
    </Modal>
  );
}
