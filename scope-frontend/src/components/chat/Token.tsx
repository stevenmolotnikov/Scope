'use client';

import { useCallback, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  formatTokenForDisplay,
  determineTokenClass,
  getColorForProbability,
  getColorForRank,
  formatProbability,
  cleanTokenizerArtifacts,
} from '@/lib/utils';
import { useUIStore } from '@/stores/uiStore';
import type { Token as TokenType, SelectedTokenInfo, TokenAlternative } from '@/types';

interface TokenProps {
  token: TokenType;
  messageId: string;
  tokenIndex: number;
}

// Get color for diff view based on probability difference (already in percentage points)
function getColorForDiff(probDiff: number | undefined): string {
  if (probDiff === undefined) return 'rgba(200, 200, 200, 0.3)';
  
  // Positive diff = gen model had higher prob (good for gen)
  // Negative diff = analysis model had higher prob (bad for gen)
  // probDiff is in percentage points (e.g., 10 means 10% difference)
  if (probDiff > 10) return 'rgba(34, 197, 94, 0.6)'; // Strong green
  if (probDiff > 1) return 'rgba(134, 239, 172, 0.5)'; // Light green
  if (probDiff < -10) return 'rgba(239, 68, 68, 0.6)'; // Strong red
  if (probDiff < -1) return 'rgba(252, 165, 165, 0.5)'; // Light red
  return 'rgba(200, 200, 200, 0.3)'; // Neutral gray
}

export function Token({ token, messageId, tokenIndex }: TokenProps) {
  const tokenRef = useRef<HTMLSpanElement>(null);
  const [isHovered, setIsHovered] = useState(false);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  const {
    viewMode,
    highlightMode,
    selectedToken,
    selectToken,
    setHoveredToken,
  } = useUIStore();

  const cleanToken = cleanTokenizerArtifacts(token.token);
  const tokenClass = determineTokenClass(token.token);
  const isSelected =
    selectedToken?.messageId === messageId && selectedToken?.tokenIndex === tokenIndex;

  // Determine token type
  const isNewline = tokenClass === 'newline-token' || cleanToken.includes('\n');
  const isSpace = cleanToken === ' ' || /^\s+$/.test(cleanToken);

  // Calculate background color based on view mode
  let backgroundColor: string;
  if (viewMode === 'text') {
    backgroundColor = 'transparent';
  } else if (viewMode === 'diff') {
    backgroundColor = getColorForDiff(token.diff_data?.prob_diff);
  } else {
    backgroundColor = highlightMode === 'rank'
      ? getColorForRank(token.rank, token.vocab_size)
      : getColorForProbability(token.probability);
  }

  const handleClick = useCallback(() => {
    if (isSelected) {
      selectToken(null);
    } else {
      const info: SelectedTokenInfo = {
        token,
        messageId,
        tokenIndex,
        element: tokenRef.current || undefined,
      };
      selectToken(info);
    }
  }, [token, messageId, tokenIndex, isSelected, selectToken]);

  const handleMouseEnter = useCallback((e: React.MouseEvent) => {
    setIsHovered(true);
    setHoveredToken({ token, messageId, tokenIndex });
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    setTooltipPos({
      x: rect.left + rect.width / 2,
      y: rect.top,
    });
  }, [token, messageId, tokenIndex, setHoveredToken]);

  const handleMouseLeave = useCallback(() => {
    setIsHovered(false);
    setHoveredToken(null);
  }, [setHoveredToken]);

  // Display text - use symbol (⎵) for space tokens in token view
  const displayText = formatTokenForDisplay(token.token);
  const tooltipDisplay = displayText;

  // Text view mode - plain text with hover border
  if (viewMode === 'text') {
    return (
      <>
        <span
          ref={tokenRef}
          onClick={handleClick}
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
          className="font-mono"
          style={{
            cursor: 'pointer',
            whiteSpace: 'pre',
            borderRadius: '2px',
            border: isSelected ? '1px solid #000' : '1px solid transparent',
            padding: '0 1px',
            margin: '0 -1px',
            background: isSelected ? 'rgba(0,0,0,0.05)' : 'transparent',
            transition: 'border-color 0.1s, background 0.1s',
          }}
          onMouseOver={(e) => {
            if (!isSelected) {
              e.currentTarget.style.borderColor = '#999';
              e.currentTarget.style.background = 'rgba(0,0,0,0.03)';
            }
          }}
          onMouseOut={(e) => {
            if (!isSelected) {
              e.currentTarget.style.borderColor = 'transparent';
              e.currentTarget.style.background = 'transparent';
            }
          }}
        >
          {cleanToken}
        </span>
        {isNewline && <br />}
      </>
    );
  }

  // Token/Diff view mode - styled tokens
  return (
    <>
      <span
        ref={tokenRef}
        onClick={handleClick}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        className="font-mono"
        style={{
          display: 'inline-block',
          padding: '1px 3px',
          margin: '1px',
          borderRadius: '3px',
          backgroundColor,
          cursor: 'pointer',
          border: isSelected 
            ? '1px solid #000' 
            : token.rule_applied 
            ? '1px dashed #666' 
            : '1px solid transparent',
          boxShadow: isSelected ? '0 0 0 1px rgba(0,0,0,0.15)' : undefined,
          fontSize: '13px',
          lineHeight: '1.5',
          whiteSpace: 'pre',
          minWidth: isSpace ? '0.5em' : undefined,
          position: 'relative',
          transition: 'border-color 0.1s, box-shadow 0.1s',
        }}
        onMouseOver={(e) => {
          if (!isSelected) {
            e.currentTarget.style.borderColor = '#000';
            e.currentTarget.style.boxShadow = '0 1px 4px rgba(0,0,0,0.1)';
          }
        }}
        onMouseOut={(e) => {
          if (!isSelected) {
            e.currentTarget.style.borderColor = token.rule_applied ? '#666' : 'transparent';
            e.currentTarget.style.boxShadow = 'none';
          }
        }}
      >
        {displayText}

        {/* Rule applied indicator */}
        {token.rule_applied && (
          <span style={{
            position: 'absolute',
            top: '-4px',
            right: '-4px',
            fontSize: '8px',
            background: '#fff',
            borderRadius: '50%',
            border: '1px solid #ccc',
            width: '12px',
            height: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            ⚙
          </span>
        )}
      </span>

      {/* Hover tooltip */}
      {isHovered && token.top_alternatives && token.top_alternatives.length > 0 && typeof document !== 'undefined' && createPortal(
        <div
          style={{
            position: 'fixed',
            left: tooltipPos.x,
            top: tooltipPos.y - 8,
            transform: 'translate(-50%, -100%)',
            zIndex: 10000,
            pointerEvents: 'none',
          }}
        >
          <div 
            className="font-mono"
            style={{
              background: '#fff',
              border: '1px solid #ddd',
              borderRadius: '6px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              padding: '8px 12px',
              minWidth: '180px',
              maxWidth: '280px',
              fontSize: '12px',
            }}
          >
            {/* Token header */}
            <div style={{
              marginBottom: '6px',
              paddingBottom: '6px',
              borderBottom: '1px solid #eee',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}>
              <span style={{ fontWeight: 600 }}>"{tooltipDisplay}"</span>
              <span style={{ color: '#666' }}>
                {formatProbability(token.probability)}
              </span>
            </div>

            {/* Diff info when in diff mode or has diff data */}
            {token.diff_data && (
              <div style={{
                marginBottom: '6px',
                paddingBottom: '6px',
                borderBottom: '1px solid #eee',
                fontSize: '11px',
              }}>
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  color: '#666',
                  marginBottom: '2px',
                }}>
                  <span>Gen Model:</span>
                  <span>{formatProbability(token.probability)}</span>
                </div>
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  color: '#666',
                  marginBottom: '2px',
                }}>
                  <span>Analysis Model:</span>
                  <span>{formatProbability(token.diff_data.analysis_prob)}</span>
                </div>
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  fontWeight: 600,
                  color: token.diff_data.prob_diff > 0 ? '#16a34a' : token.diff_data.prob_diff < 0 ? '#dc2626' : '#666',
                }}>
                  <span>Diff:</span>
                  <span>{token.diff_data.prob_diff > 0 ? '+' : ''}{token.diff_data.prob_diff.toFixed(1)}%</span>
                </div>
              </div>
            )}

            {/* Top alternatives - show both models if diff data exists */}
            {token.diff_data ? (
              <div style={{ fontSize: '11px' }}>
                {/* Gen model alternatives */}
                <div style={{ color: '#666', marginBottom: '6px' }}>
                  <div style={{ fontWeight: 600, marginBottom: '4px', color: '#333' }}>Gen Model:</div>
                  {token.top_alternatives.slice(0, 3).map((alt: TokenAlternative, idx: number) => (
                    <div key={idx} style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between',
                      padding: '2px 0',
                    }}>
                      <span>#{alt.rank} "{formatTokenForDisplay(alt.token)}"</span>
                      <span>{formatProbability(alt.probability)}</span>
                    </div>
                  ))}
                </div>
                {/* Analysis model alternatives */}
                <div style={{ color: '#666', paddingTop: '6px', borderTop: '1px solid #eee' }}>
                  <div style={{ fontWeight: 600, marginBottom: '4px', color: '#333' }}>Analysis Model:</div>
                  {token.diff_data.analysis_top_alternatives.slice(0, 3).map((alt: TokenAlternative, idx: number) => (
                    <div key={idx} style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between',
                      padding: '2px 0',
                    }}>
                      <span>#{alt.rank} "{formatTokenForDisplay(alt.token)}"</span>
                      <span>{formatProbability(alt.probability)}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ color: '#666', fontSize: '11px' }}>
                {token.top_alternatives.slice(0, 3).map((alt: TokenAlternative, idx: number) => (
                  <div key={idx} style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between',
                    padding: '2px 0',
                  }}>
                    <span>#{alt.rank} "{formatTokenForDisplay(alt.token)}"</span>
                    <span>{formatProbability(alt.probability)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>,
        document.body
      )}

      {/* Add line break after newline tokens */}
      {isNewline && <br />}
    </>
  );
}
