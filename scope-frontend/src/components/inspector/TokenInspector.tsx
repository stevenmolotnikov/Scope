'use client';

import { useState, useCallback, useEffect } from 'react';
import { useUIStore } from '@/stores/uiStore';
import { useConversationStore } from '@/stores/conversationStore';
import { useStreamingGeneration } from '@/hooks/useStreamingGeneration';
import { formatTokenForDisplay, formatProbability } from '@/lib/utils';
import type { TokenAlternative, SelectedTokenInfo } from '@/types';

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

  const handleLogitLens = useCallback(() => {
    if (!selectedToken || !conversation) return;
    const message = conversation.messageTree[selectedToken.messageId];
    if (message?.tokens) {
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
          }}
        >
          ✕
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
          <div>
            {/* Token Header */}
            <div style={{
              background: '#f8f8f8',
              border: '1px solid #e5e5e5',
              borderRadius: '8px',
              padding: '12px',
              marginBottom: '12px',
            }}>
              <div className="font-mono" style={{
                fontSize: '18px',
                fontWeight: 600,
                marginBottom: '8px',
                wordBreak: 'break-all',
              }}>
                "{formatTokenForDisplay(token.token)}"
              </div>
              
              {/* Stats Grid */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '8px',
                fontSize: '11px',
              }}>
                <div>
                  <div style={{ color: '#999' }}>{viewMode === 'diff' && token.diff_data ? 'Gen Prob' : 'Probability'}</div>
                  <div className="font-mono" style={{ fontWeight: 600 }}>
                    {formatProbability(token.probability)}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#999' }}>{viewMode === 'diff' && token.diff_data ? 'Gen Rank' : 'Rank'}</div>
                  <div className="font-mono" style={{ fontWeight: 600 }}>
                    #{token.rank ?? 'N/A'}
                  </div>
                </div>
              </div>

              {/* Analysis Model Stats - only in diff view */}
              {viewMode === 'diff' && token.diff_data && (
                <>
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: '8px',
                    fontSize: '11px',
                    marginTop: '8px',
                    paddingTop: '8px',
                    borderTop: '1px solid #e5e5e5',
                  }}>
                    <div>
                      <div style={{ color: '#999' }}>Analysis Prob</div>
                      <div className="font-mono" style={{ fontWeight: 600 }}>
                        {formatProbability(token.diff_data.analysis_prob)}
                      </div>
                    </div>
                    <div>
                      <div style={{ color: '#999' }}>Analysis Rank</div>
                      <div className="font-mono" style={{ fontWeight: 600 }}>
                        #{token.diff_data.analysis_rank ?? 'N/A'}
                      </div>
                    </div>
                  </div>
                  
                  {/* Diff Badge */}
                  <div style={{
                    marginTop: '10px',
                    padding: '6px 10px',
                    borderRadius: '6px',
                    background: token.diff_data.prob_diff > 0 ? '#dcfce7' : token.diff_data.prob_diff < 0 ? '#fee2e2' : '#f5f5f5',
                    border: `1px solid ${token.diff_data.prob_diff > 0 ? '#86efac' : token.diff_data.prob_diff < 0 ? '#fca5a5' : '#e5e5e5'}`,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}>
                    <span style={{ fontSize: '11px', color: '#666' }}>Diff</span>
                    <span className="font-mono" style={{ 
                      fontSize: '12px',
                      fontWeight: 600,
                      color: token.diff_data.prob_diff > 0 ? '#16a34a' : token.diff_data.prob_diff < 0 ? '#dc2626' : '#666',
                    }}>
                      {token.diff_data.prob_diff > 0 ? '+' : ''}{token.diff_data.prob_diff.toFixed(1)}%
                    </span>
                  </div>
                </>
              )}
              
              {/* Logit Lens Button */}
              <button
                onClick={handleLogitLens}
                style={{
                  marginTop: '10px',
                  width: '100%',
                  padding: '8px 10px',
                  background: '#f0f0f0',
                  border: '1px solid #ddd',
                  borderRadius: '6px',
                  fontSize: '12px',
                  fontWeight: 500,
                  cursor: 'pointer',
                  color: '#333',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = '#e8e8e8';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = '#f0f0f0';
                }}
              >
                🔬 Logit Lens
              </button>
            </div>

            {/* Gen Model Alternatives */}
            {token.top_alternatives && token.top_alternatives.length > 0 && (
              <div style={{ marginBottom: '12px' }}>
                <h4 style={{
                  fontSize: '11px',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  color: '#999',
                  marginBottom: '8px',
                }}>
                  {token.diff_data && viewMode === 'diff' ? 'Gen Model Alternatives' : 'Alternatives'}
                </h4>
                <div style={{ 
                  border: '1px solid #e5e5e5', 
                  borderRadius: '8px',
                  overflow: 'hidden',
                }}>
                  {token.top_alternatives.slice(0, 3).map((alt: TokenAlternative, idx: number) => {
                    const isAltSelected = selectedAltIndex === idx;
                    return (
                      <div
                        key={idx}
                        onClick={() => handleSelectAlternative(alt, idx)}
                        className="font-mono"
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          padding: '8px 10px',
                          background: isAltSelected ? '#e0e0e0' : (idx % 2 === 0 ? '#fafafa' : '#fff'),
                          fontSize: '12px',
                          cursor: 'pointer',
                          borderBottom: idx < 2 ? '1px solid #eee' : 'none',
                        }}
                        onMouseOver={(e) => {
                          if (!isAltSelected) {
                            e.currentTarget.style.background = '#f0f0f0';
                          }
                        }}
                        onMouseOut={(e) => {
                          if (!isAltSelected) {
                            e.currentTarget.style.background = idx % 2 === 0 ? '#fafafa' : '#fff';
                          }
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ color: '#999', fontSize: '10px', width: '20px' }}>
                            #{alt.rank}
                          </span>
                          <span style={{ fontWeight: isAltSelected ? 600 : 400 }}>
                            "{formatTokenForDisplay(alt.token)}"
                          </span>
                        </div>
                        <span style={{ color: '#666', fontSize: '11px' }}>
                          {formatProbability(alt.probability)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Analysis Model Alternatives (only in diff view) */}
            {viewMode === 'diff' && token.diff_data && token.diff_data.analysis_top_alternatives && token.diff_data.analysis_top_alternatives.length > 0 && (
              <div style={{ marginBottom: '12px' }}>
                <h4 style={{
                  fontSize: '11px',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  color: '#999',
                  marginBottom: '8px',
                }}>
                  Analysis Model Alternatives
                </h4>
                <div style={{ 
                  border: '1px solid #e5e5e5', 
                  borderRadius: '8px',
                  overflow: 'hidden',
                }}>
                  {token.diff_data.analysis_top_alternatives.slice(0, 3).map((alt: TokenAlternative, idx: number) => (
                    <div
                      key={idx}
                      onClick={() => handleSelectAlternative(alt, idx + 100)} // Offset to distinguish from gen alts
                      className="font-mono"
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '8px 10px',
                        background: selectedAltIndex === idx + 100 ? '#e0e0e0' : (idx % 2 === 0 ? '#fafafa' : '#fff'),
                        fontSize: '12px',
                        cursor: 'pointer',
                        borderBottom: idx < 2 ? '1px solid #eee' : 'none',
                      }}
                      onMouseOver={(e) => {
                        if (selectedAltIndex !== idx + 100) {
                          e.currentTarget.style.background = '#f0f0f0';
                        }
                      }}
                      onMouseOut={(e) => {
                        if (selectedAltIndex !== idx + 100) {
                          e.currentTarget.style.background = idx % 2 === 0 ? '#fafafa' : '#fff';
                        }
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ color: '#999', fontSize: '10px', width: '20px' }}>
                          #{alt.rank}
                        </span>
                        <span style={{ fontWeight: selectedAltIndex === idx + 100 ? 600 : 400 }}>
                          "{formatTokenForDisplay(alt.token)}"
                        </span>
                      </div>
                      <span style={{ color: '#666', fontSize: '11px' }}>
                        {formatProbability(alt.probability)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Rule Applied */}
            {token.rule_applied && (
              <div style={{
                background: '#fff8e0',
                border: '1px solid #f0d860',
                borderRadius: '8px',
                padding: '12px',
                marginBottom: '12px',
                fontSize: '12px',
              }}>
                <div style={{ fontWeight: 600, marginBottom: '4px' }}>
                  ⚙ {token.rule_applied.name}
                </div>
                <div style={{ color: '#666' }}>{token.rule_applied.reason}</div>
              </div>
            )}

            {/* Injection */}
            <div>
              <h4 style={{
                fontSize: '11px',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                color: '#999',
                marginBottom: '8px',
              }}>
                Inject
              </h4>
              <div style={{ display: 'flex', gap: '6px' }}>
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
                    borderRadius: '6px',
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
                    borderRadius: '6px',
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
