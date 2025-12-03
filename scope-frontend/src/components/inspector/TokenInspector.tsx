'use client';

import { useState, useCallback, useEffect } from 'react';
import { useUIStore } from '@/stores/uiStore';
import { useConversationStore } from '@/stores/conversationStore';
import { useStreamingGeneration } from '@/hooks/useStreamingGeneration';
import { formatTokenForDisplay, formatProbability, formatModelName } from '@/lib/utils';
import type { TokenAlternative, SelectedTokenInfo } from '@/types';
import { X, Search } from 'lucide-react';

// Section header component
function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      fontSize: '10px',
      fontWeight: 600,
      textTransform: 'uppercase',
      letterSpacing: '0.05em',
      color: '#666',
      padding: '8px 10px',
      borderBottom: '1px solid #e5e5e5',
      background: '#fafafa',
      fontFamily: 'monospace',
    }}>
      {children}
    </div>
  );
}

// Alternatives list component
function AlternativesList({ 
  alternatives, 
  selectedIndex, 
  onSelect,
  indexOffset = 0,
}: { 
  alternatives: TokenAlternative[];
  selectedIndex: number | null;
  onSelect: (alt: TokenAlternative, index: number) => void;
  indexOffset?: number;
}) {
  return (
    <div style={{ borderBottom: '1px solid #e5e5e5' }}>
      {alternatives.slice(0, 3).map((alt, idx) => {
        const actualIndex = idx + indexOffset;
        const isSelected = selectedIndex === actualIndex;
        return (
          <div
            key={idx}
            onClick={() => onSelect(alt, actualIndex)}
            className="font-mono"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '6px 10px',
              background: isSelected ? '#e0e0e0' : '#fff',
              fontSize: '11px',
              cursor: 'pointer',
              borderBottom: idx < 2 ? '1px solid #f0f0f0' : 'none',
            }}
            onMouseOver={(e) => {
              if (!isSelected) e.currentTarget.style.background = '#f5f5f5';
            }}
            onMouseOut={(e) => {
              if (!isSelected) e.currentTarget.style.background = '#fff';
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ color: '#999', fontSize: '9px', width: '18px' }}>#{alt.rank}</span>
              <span style={{ fontWeight: isSelected ? 600 : 400 }}>"{formatTokenForDisplay(alt.token)}"</span>
            </div>
            <span style={{ color: '#666' }}>{formatProbability(alt.probability)}</span>
          </div>
        );
      })}
    </div>
  );
}

export function TokenInspector() {
  const { 
    rightSidebarCollapsed, 
    selectedToken, 
    toggleRightSidebar,
    setLogitLensContext,
    persistedTokenRef,
    selectToken,
    viewMode,
  } = useUIStore();
  const { getCurrentConversation } = useConversationStore();
  const { injectAndRegenerate } = useStreamingGeneration();
  const [injectionText, setInjectionText] = useState('');
  const [selectedAltIndex, setSelectedAltIndex] = useState<number | null>(null);

  const conversation = getCurrentConversation();
  const token = selectedToken?.token;
  const isDiffView = viewMode === 'diff' && token?.diff_data;

  // Rehydrate selectedToken from persistedTokenRef when conversation loads
  useEffect(() => {
    if (persistedTokenRef && conversation && !selectedToken) {
      const message = conversation.messageTree[persistedTokenRef.messageId];
      if (message?.tokens && message.tokens[persistedTokenRef.tokenIndex]) {
        const tokenData = message.tokens[persistedTokenRef.tokenIndex];
        const info: SelectedTokenInfo = {
          token: tokenData,
          messageId: persistedTokenRef.messageId,
          tokenIndex: persistedTokenRef.tokenIndex,
        };
        selectToken(info);
      }
    }
  }, [conversation, persistedTokenRef, selectedToken, selectToken]);

  // Reset selected alternative when token changes
  useEffect(() => {
    setSelectedAltIndex(null);
    setInjectionText('');
  }, [selectedToken]);

  const handleSelectAlternative = useCallback((alt: TokenAlternative, index: number) => {
    setSelectedAltIndex(index);
    setInjectionText(alt.token);
  }, []);

  const handleInject = useCallback(() => {
    if (!selectedToken || !injectionText.trim()) return;
    injectAndRegenerate(selectedToken.messageId, selectedToken.tokenIndex, injectionText, {
      onError: (error) => console.error('Injection error:', error),
    });
    setInjectionText('');
    setSelectedAltIndex(null);
  }, [selectedToken, injectionText, injectAndRegenerate]);

  const handleLogitLens = useCallback((model?: string) => {
    if (!selectedToken || !conversation) return;
    const message = conversation.messageTree[selectedToken.messageId];
    if (message?.tokens) {
      // TODO: Pass model to logit lens context if needed for model-specific analysis
      setLogitLensContext(message.tokens, selectedToken.messageId, selectedToken.tokenIndex);
    }
  }, [selectedToken, conversation, setLogitLensContext]);

  if (rightSidebarCollapsed) {
    return null;
  }

  return (
    <div style={{
      width: '300px',
      flexShrink: 0,
      display: 'flex',
      flexDirection: 'column',
      background: '#fff',
      borderLeft: '1px solid #e5e5e5',
      height: '100%',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '12px 16px',
        borderBottom: '1px solid #e5e5e5',
      }}>
        <h3 style={{
          fontSize: '12px',
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          color: '#666',
          margin: 0,
        }}>
          Inspector
        </h3>
        <button
          onClick={toggleRightSidebar}
          style={{
            padding: '4px',
            background: 'transparent',
            border: 'none',
            cursor: 'pointer',
            color: '#999',
            borderRadius: '4px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <X size={16} />
        </button>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'auto', padding: '16px' }}>
        {!token ? (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '200px',
            color: '#999',
            textAlign: 'center',
          }}>
            <p style={{ fontSize: '13px' }}>Click any token to inspect</p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            
            {/* ==================== TOKEN BOX ==================== */}
            <div style={{
              background: '#f8f8f8',
              border: '1px solid #e5e5e5',
              borderRadius: '8px',
              padding: '16px',
              textAlign: 'center',
            }}>
              {/* Token Display */}
              <div className="font-mono" style={{
                fontSize: '22px',
                fontWeight: 600,
                wordBreak: 'break-all',
                marginBottom: '12px',
              }}>
                "{formatTokenForDisplay(token.token)}"
              </div>

              {/* Standard View: Show Prob + Rank */}
              {!isDiffView && (
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '8px',
                  fontSize: '12px',
                }}>
                  <div>
                    <div style={{ color: '#999', fontSize: '10px', marginBottom: '2px' }}>Probability</div>
                    <div className="font-mono" style={{ fontWeight: 600 }}>
                      {formatProbability(token.probability)}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#999', fontSize: '10px', marginBottom: '2px' }}>Rank</div>
                    <div className="font-mono" style={{ fontWeight: 600 }}>
                      #{token.rank ?? 'N/A'}
                    </div>
                  </div>
                </div>
              )}

              {/* Diff View: Show Diff Badge */}
              {isDiffView && token.diff_data && (
                <div style={{
                  padding: '10px 16px',
                  borderRadius: '6px',
                  background: token.diff_data.prob_diff > 0 ? '#dcfce7' : token.diff_data.prob_diff < 0 ? '#fee2e2' : '#f5f5f5',
                  border: `1px solid ${token.diff_data.prob_diff > 0 ? '#86efac' : token.diff_data.prob_diff < 0 ? '#fca5a5' : '#e5e5e5'}`,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}>
                  <span style={{ fontSize: '12px', color: '#666' }}>Diff</span>
                  <span className="font-mono" style={{ 
                    fontSize: '14px',
                    fontWeight: 600,
                    color: token.diff_data.prob_diff > 0 ? '#16a34a' : token.diff_data.prob_diff < 0 ? '#dc2626' : '#666',
                  }}>
                    {token.diff_data.prob_diff > 0 ? '+' : ''}{token.diff_data.prob_diff.toFixed(1)}%
                  </span>
                </div>
              )}
            </div>

            {/* ==================== STANDARD VIEW: MODEL BOX ==================== */}
            {!isDiffView && (
              <div style={{
                border: '1px solid #e5e5e5',
                borderRadius: '8px',
                overflow: 'hidden',
              }}>
                {/* Model Header */}
                <SectionHeader>
                  {formatModelName(conversation?.model || '')}
                </SectionHeader>

                {/* Alternatives */}
                {token.top_alternatives && token.top_alternatives.length > 0 && (
                  <AlternativesList
                    alternatives={token.top_alternatives}
                    selectedIndex={selectedAltIndex}
                    onSelect={handleSelectAlternative}
                    indexOffset={0}
                  />
                )}

                {/* LogitLens Button */}
                <div style={{ padding: '8px 10px' }}>
                  <button
                    onClick={() => handleLogitLens()}
                    style={{
                      width: '100%',
                      padding: '8px',
                      background: '#f5f5f5',
                      border: '1px solid #ddd',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: 500,
                      cursor: 'pointer',
                      color: '#555',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '6px',
                    }}
                    onMouseOver={(e) => e.currentTarget.style.background = '#eee'}
                    onMouseOut={(e) => e.currentTarget.style.background = '#f5f5f5'}
                  >
                    <Search size={12} /> LogitLens
                  </button>
                </div>
              </div>
            )}

            {/* ==================== DIFF VIEW: GENERATION BOX ==================== */}
            {isDiffView && (
              <div style={{
                border: '1px solid #e5e5e5',
                borderRadius: '8px',
                overflow: 'hidden',
              }}>
                {/* Header */}
                <SectionHeader>
                  Generation · {formatModelName(conversation?.model || '')}
                </SectionHeader>

                {/* Stats */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '8px',
                  padding: '10px',
                  borderBottom: '1px solid #e5e5e5',
                  fontSize: '11px',
                }}>
                  <div>
                    <div style={{ color: '#999', fontSize: '9px', marginBottom: '2px' }}>Probability</div>
                    <div className="font-mono" style={{ fontWeight: 600 }}>
                      {formatProbability(token.probability)}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#999', fontSize: '9px', marginBottom: '2px' }}>Rank</div>
                    <div className="font-mono" style={{ fontWeight: 600 }}>
                      #{token.rank ?? 'N/A'}
                    </div>
                  </div>
                </div>

                {/* Alternatives */}
                {token.top_alternatives && token.top_alternatives.length > 0 && (
                  <AlternativesList
                    alternatives={token.top_alternatives}
                    selectedIndex={selectedAltIndex}
                    onSelect={handleSelectAlternative}
                    indexOffset={0}
                  />
                )}

                {/* LogitLens Button */}
                <div style={{ padding: '8px 10px' }}>
                  <button
                    onClick={() => handleLogitLens(conversation?.model)}
                    style={{
                      width: '100%',
                      padding: '8px',
                      background: '#f5f5f5',
                      border: '1px solid #ddd',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: 500,
                      cursor: 'pointer',
                      color: '#555',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '6px',
                    }}
                    onMouseOver={(e) => e.currentTarget.style.background = '#eee'}
                    onMouseOut={(e) => e.currentTarget.style.background = '#f5f5f5'}
                  >
                    <Search size={12} /> LogitLens
                  </button>
                </div>
              </div>
            )}

            {/* ==================== DIFF VIEW: ANALYSIS BOX ==================== */}
            {isDiffView && token.diff_data && (
              <div style={{
                border: '1px solid #e5e5e5',
                borderRadius: '8px',
                overflow: 'hidden',
              }}>
                {/* Header */}
                <SectionHeader>
                  Analysis · {formatModelName(token.diff_data.analysis_model || '')}
                </SectionHeader>

                {/* Stats */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '8px',
                  padding: '10px',
                  borderBottom: '1px solid #e5e5e5',
                  fontSize: '11px',
                }}>
                  <div>
                    <div style={{ color: '#999', fontSize: '9px', marginBottom: '2px' }}>Probability</div>
                    <div className="font-mono" style={{ fontWeight: 600 }}>
                      {formatProbability(token.diff_data.analysis_prob)}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#999', fontSize: '9px', marginBottom: '2px' }}>Rank</div>
                    <div className="font-mono" style={{ fontWeight: 600 }}>
                      #{token.diff_data.analysis_rank ?? 'N/A'}
                    </div>
                  </div>
                </div>

                {/* Alternatives */}
                {token.diff_data.analysis_top_alternatives && token.diff_data.analysis_top_alternatives.length > 0 && (
                  <AlternativesList
                    alternatives={token.diff_data.analysis_top_alternatives}
                    selectedIndex={selectedAltIndex}
                    onSelect={handleSelectAlternative}
                    indexOffset={100}
                  />
                )}

                {/* LogitLens Button */}
                <div style={{ padding: '8px 10px' }}>
                  <button
                    onClick={() => handleLogitLens(token.diff_data?.analysis_model)}
                    style={{
                      width: '100%',
                      padding: '8px',
                      background: '#f5f5f5',
                      border: '1px solid #ddd',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: 500,
                      cursor: 'pointer',
                      color: '#555',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '6px',
                    }}
                    onMouseOver={(e) => e.currentTarget.style.background = '#eee'}
                    onMouseOut={(e) => e.currentTarget.style.background = '#f5f5f5'}
                  >
                    <Search size={12} /> LogitLens
                  </button>
                </div>
              </div>
            )}

            {/* ==================== RULE APPLIED ==================== */}
            {token.rule_applied && (
              <div style={{
                background: '#fff8e0',
                border: '1px solid #f0d860',
                borderRadius: '8px',
                padding: '10px',
                fontSize: '11px',
              }}>
                <div style={{ fontWeight: 600, marginBottom: '4px' }}>
                  ⚙ {token.rule_applied.name}
                </div>
                <div style={{ color: '#666' }}>{token.rule_applied.reason}</div>
              </div>
            )}

            {/* ==================== INJECT ==================== */}
            <div style={{
              border: '1px solid #e5e5e5',
              borderRadius: '8px',
              overflow: 'hidden',
            }}>
              <SectionHeader>Inject</SectionHeader>
              <div style={{ padding: '10px', display: 'flex', gap: '6px' }}>
                <input
                  type="text"
                  value={injectionText}
                  onChange={(e) => {
                    setInjectionText(e.target.value);
                    setSelectedAltIndex(null);
                  }}
                  placeholder="Text to inject..."
                  className="font-mono"
                  style={{
                    flex: 1,
                    padding: '8px 10px',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    fontSize: '12px',
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleInject();
                  }}
                />
                <button
                  onClick={handleInject}
                  disabled={!injectionText.trim()}
                  style={{
                    padding: '8px 12px',
                    background: injectionText.trim() ? '#000' : '#e5e5e5',
                    color: injectionText.trim() ? '#fff' : '#999',
                    border: 'none',
                    borderRadius: '4px',
                    fontSize: '12px',
                    fontWeight: 500,
                    cursor: injectionText.trim() ? 'pointer' : 'default',
                  }}
                >
                  Go
                </button>
              </div>
            </div>

          </div>
        )}
      </div>
    </div>
  );
}
