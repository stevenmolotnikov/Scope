'use client';

import { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui';
import { useUIStore } from '@/stores/uiStore';
import { useConversationStore } from '@/stores/conversationStore';
import { api } from '@/lib/api';
import { getColorForProbability, cleanTokenizerArtifacts } from '@/lib/utils';
import type { LogitLensResponse, Token } from '@/types';

type ViewType = 'heatmap' | 'chart';

// Display token - just make whitespace visible, show everything else as-is
function safeTokenDisplay(token: string, maxLen: number = 8): string {
  const clean = cleanTokenizerArtifacts(token);
  
  const display = clean
    .replace(/\n/g, '↵')
    .replace(/\t/g, '→')
    .replace(/\r/g, '')
    .replace(/ /g, '⎵');
  
  if (display.length === 0) return '∅';
  if (display.length > maxLen) return display.slice(0, maxLen) + '…';
  return display;
}

export function LogitLensModal() {
  const [view, setView] = useState<ViewType>('heatmap');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<LogitLensResponse | null>(null);
  const [windowTokens, setWindowTokens] = useState<Token[]>([]);

  const { 
    modals, 
    closeModal, 
    logitLensTokens, 
    logitLensTokenIndex,
    clearLogitLensContext,
  } = useUIStore();
  const { getCurrentConversation, getConversationHistory } = useConversationStore();

  const isOpen = modals.logitLens;

  // Fetch logit lens data when modal opens
  useEffect(() => {
    if (!isOpen || logitLensTokens.length === 0) return;

    const fetchData = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const conversation = getCurrentConversation();
        if (!conversation) throw new Error('No conversation');

        // Analyze ALL tokens up to and including the selected token
        const endIdx = logitLensTokenIndex + 1;
        const tokens = logitLensTokens.slice(0, endIdx);
        setWindowTokens(tokens);
        
        const contextTokens: Token[] = [];

        if (tokens.length === 0) {
          throw new Error('No tokens to analyze');
        }

        const response = await api.logitLens({
          model: conversation.model,
          context: getConversationHistory().slice(0, -1).map(msg => ({
            role: msg.role as 'user' | 'assistant' | 'system',
            content: msg.content,
          })),
          context_tokens: contextTokens,
          window_tokens: tokens,
          top_k: 20,
        });

        if (response.error) {
          throw new Error(response.error);
        }

        setData(response);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load logit lens data');
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [isOpen, logitLensTokens, logitLensTokenIndex, getCurrentConversation, getConversationHistory]);

  const handleClose = useCallback(() => {
    closeModal('logitLens');
    clearLogitLensContext();
    setData(null);
    setError(null);
    setWindowTokens([]);
  }, [closeModal, clearLogitLensContext]);

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      zIndex: 1000,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    }}>
      {/* Backdrop */}
      <div 
        onClick={handleClose}
        style={{
          position: 'absolute',
          inset: 0,
          background: 'rgba(0,0,0,0.5)',
        }}
      />

      {/* Modal */}
      <div style={{
        position: 'relative',
        width: '90vw',
        maxWidth: '1200px',
        height: '80vh',
        background: '#fff',
        borderRadius: '12px',
        boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{
          padding: '16px 24px',
          borderBottom: '1px solid #e5e5e5',
          background: '#fafafa',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <h2 style={{ margin: 0, fontSize: '16px', fontWeight: 600 }}>
            🔬 Logit Lens Analysis
          </h2>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {/* View toggle */}
            <div style={{
              display: 'flex',
              gap: '4px',
              background: '#fff',
              padding: '4px',
              borderRadius: '8px',
              border: '1px solid #e5e5e5',
            }}>
              <button
                onClick={() => setView('heatmap')}
                style={{
                  padding: '6px 12px',
                  fontSize: '13px',
                  fontWeight: 500,
                  borderRadius: '6px',
                  border: 'none',
                  cursor: 'pointer',
                  background: view === 'heatmap' ? '#000' : 'transparent',
                  color: view === 'heatmap' ? '#fff' : '#666',
                }}
              >
                Heatmap
              </button>
              <button
                onClick={() => setView('chart')}
                style={{
                  padding: '6px 12px',
                  fontSize: '13px',
                  fontWeight: 500,
                  borderRadius: '6px',
                  border: 'none',
                  cursor: 'pointer',
                  background: view === 'chart' ? '#000' : 'transparent',
                  color: view === 'chart' ? '#fff' : '#666',
                }}
              >
                Chart
              </button>
            </div>

            <button
              onClick={handleClose}
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '6px',
                border: 'none',
                background: '#f5f5f5',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '16px',
                color: '#666',
              }}
            >
              ✕
            </button>
          </div>
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflow: 'auto', padding: '20px' }}>
          {isLoading && (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              gap: '16px',
              color: '#666',
            }}>
              <div style={{
                width: '40px',
                height: '40px',
                border: '3px solid #e5e5e5',
                borderTopColor: '#000',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite',
              }} />
              <p>Analyzing layers...</p>
              <style>{`
                @keyframes spin {
                  to { transform: rotate(360deg); }
                }
              `}</style>
            </div>
          )}

          {error && (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              gap: '16px',
              color: '#c00',
            }}>
              <p>Error: {error}</p>
              <Button variant="secondary" onClick={handleClose}>
                Close
              </Button>
            </div>
          )}

          {!isLoading && !error && data && (
            <div style={{ height: '100%' }}>
              {view === 'heatmap' && <HeatmapView data={data} windowTokens={windowTokens} />}
              {view === 'chart' && <ChartView data={data} windowTokens={windowTokens} />}
            </div>
          )}

          {!isLoading && !error && !data && logitLensTokens.length === 0 && (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              color: '#999',
              textAlign: 'center',
            }}>
              <p style={{ marginBottom: '8px' }}>Select tokens to analyze</p>
              <p style={{ fontSize: '13px' }}>
                Click a token and use the Logit Lens button to see layer-by-layer predictions
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Heatmap visualization
function HeatmapView({ data, windowTokens }: { data: LogitLensResponse; windowTokens: Token[] }) {
  const [hoveredCell, setHoveredCell] = useState<{ layer: number; pos: number } | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  if (!data.positions || data.positions.length === 0) {
    return <div style={{ color: '#999', textAlign: 'center', padding: '40px' }}>No data available</div>;
  }

  const numLayers = data.num_layers;
  const positions = data.positions;

  // Sample layers
  const maxDisplayLayers = 32;
  const layerStep = Math.max(1, Math.ceil(numLayers / maxDisplayLayers));
  const displayLayers = Array.from(
    { length: Math.ceil(numLayers / layerStep) }, 
    (_, i) => i * layerStep
  ).filter(l => l < numLayers);

  return (
    <div style={{ overflowX: 'auto' }}>
      {/* Legend */}
      <div style={{ 
        marginBottom: '16px', 
        display: 'flex', 
        alignItems: 'center', 
        gap: '8px',
        fontSize: '12px',
        color: '#666',
      }}>
        <span>Probability:</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
          <span>0%</span>
          <div style={{
            width: '120px',
            height: '12px',
            background: 'linear-gradient(to right, #fff8e1, #ffecb3, #ffe082, #ffd54f, #ffca28, #ffc107)',
            borderRadius: '2px',
            border: '1px solid #e5e5e5',
          }} />
          <span>100%</span>
        </div>
        {layerStep > 1 && (
          <span style={{ marginLeft: '16px', color: '#999' }}>
            (showing every {layerStep} layers)
          </span>
        )}
      </div>

      <table style={{ 
        borderCollapse: 'collapse', 
        width: '100%',
        fontSize: '12px',
      }}>
        <thead>
          <tr>
            <th style={{ 
              padding: '8px', 
              textAlign: 'left',
              borderBottom: '2px solid #e5e5e5',
              background: '#fff',
              fontWeight: 600,
              position: 'sticky',
              left: 0,
              zIndex: 2,
            }}>
              Layer
            </th>
            {positions.map((pos, idx) => (
              <th 
                key={idx}
                className="font-mono"
                style={{ 
                  padding: '8px', 
                  textAlign: 'center',
                  borderBottom: '2px solid #e5e5e5',
                  fontWeight: 600,
                  minWidth: '80px',
                }}
                title={windowTokens[idx]?.token || `Position ${pos.position}`}
              >
                {windowTokens[idx] 
                  ? `"${safeTokenDisplay(windowTokens[idx].token, 8)}"` 
                  : `Pos ${pos.position}`}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {displayLayers.map((layerIdx) => (
            <tr key={layerIdx}>
              <td style={{ 
                padding: '6px 8px',
                borderBottom: '1px solid #eee',
                background: '#fff',
                fontWeight: 500,
                position: 'sticky',
                left: 0,
                zIndex: 1,
              }}>
                {layerIdx}
              </td>
              {positions.map((pos, posIdx) => {
                const layerData = pos.layer_predictions.find(l => l.layer === layerIdx);
                const topPred = layerData?.predictions[0];
                const isHovered = hoveredCell?.layer === layerIdx && hoveredCell?.pos === posIdx;
                
                return (
                  <td 
                    key={posIdx}
                    className="font-mono"
                    onMouseEnter={(e) => {
                      setHoveredCell({ layer: layerIdx, pos: posIdx });
                      const rect = e.currentTarget.getBoundingClientRect();
                      setTooltipPos({ x: rect.left + rect.width / 2, y: rect.top });
                    }}
                    onMouseLeave={() => setHoveredCell(null)}
                    style={{ 
                      padding: '4px 8px',
                      textAlign: 'center',
                      borderBottom: '1px solid #eee',
                      background: topPred ? getColorForProbability(topPred.probability) : '#f5f5f5',
                      whiteSpace: 'nowrap',
                      cursor: 'default',
                      position: 'relative',
                      outline: isHovered ? '2px solid #333' : 'none',
                      outlineOffset: '-1px',
                    }}
                  >
                    {topPred ? safeTokenDisplay(topPred.token, 8) : '-'}
                    
                    {/* Hover tooltip */}
                    {isHovered && topPred && layerData && (
                      <div style={{
                        position: 'fixed',
                        left: tooltipPos.x,
                        top: tooltipPos.y - 8,
                        transform: 'translate(-50%, -100%)',
                        background: '#fff',
                        border: '1px solid #ddd',
                        borderRadius: '6px',
                        boxShadow: '0 4px 16px rgba(0,0,0,0.2)',
                        padding: '12px',
                        minWidth: '200px',
                        zIndex: 10000,
                        textAlign: 'left',
                        fontSize: '12px',
                        pointerEvents: 'none',
                      }}>
                        <div style={{ fontWeight: 600, marginBottom: '6px', color: '#333' }}>
                          Layer {layerIdx}
                        </div>
                        {layerData.predictions.slice(0, 5).map((pred, i) => (
                          <div key={i} style={{ 
                            display: 'flex', 
                            justifyContent: 'space-between',
                            padding: '2px 0',
                          }}>
                            <span>"{safeTokenDisplay(pred.token, 10)}"</span>
                            <span style={{ color: '#666', marginLeft: '12px' }}>
                              {(pred.probability * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Chart view - line chart showing probability evolution
function ChartView({ data, windowTokens }: { data: LogitLensResponse; windowTokens: Token[] }) {
  // Default to last position (the selected token)
  const [selectedPosition, setSelectedPosition] = useState(() => 
    data.positions ? data.positions.length - 1 : 0
  );
  const [hoveredPoint, setHoveredPoint] = useState<{ layer: number; token: string; prob: number; x: number; y: number } | null>(null);
  const [customTokens, setCustomTokens] = useState<string[]>([]);
  const [newTokenInput, setNewTokenInput] = useState('');

  if (!data.positions || data.positions.length === 0) {
    return <div style={{ color: '#999', textAlign: 'center', padding: '40px' }}>No data available</div>;
  }

  const position = data.positions[selectedPosition];
  const numLayers = data.num_layers;

  // Get top 5 tokens from the highest layer (final layer)
  const sortedLayers = [...position.layer_predictions].sort((a, b) => b.layer - a.layer);
  const finalLayer = sortedLayers[0]; // Highest layer number
  const autoTrackedTokens = finalLayer?.predictions.slice(0, 5).map(p => p.token) || [];

  // Combine auto and custom tokens
  const trackedTokens = [...new Set([...autoTrackedTokens, ...customTokens])].slice(0, 8);

  // Colors for lines
  const lineColors = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4', '#795548', '#607D8B'];

  // Build lookup for layer predictions
  const layerMap = new Map(position.layer_predictions.map(l => [l.layer, l]));

  // Build data series for each tracked token - include ALL layers
  const series = trackedTokens.map((token, idx) => {
    const points: { layer: number; prob: number }[] = [];
    for (let layer = 0; layer < numLayers; layer++) {
      const layerData = layerMap.get(layer);
      const pred = layerData?.predictions.find(p => p.token === token);
      points.push({ layer, prob: pred?.probability || 0 });
    }
    return { token, color: lineColors[idx % lineColors.length], points, isCustom: customTokens.includes(token) };
  });

  // Chart dimensions
  const chartWidth = 800;
  const chartHeight = 300;
  const padding = { top: 20, right: 20, bottom: 40, left: 50 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  // Scales
  const xScale = (layer: number) => padding.left + (layer / (numLayers - 1)) * innerWidth;
  const yScale = (prob: number) => padding.top + innerHeight - (prob * innerHeight);

  // Generate path for each series - include all points
  const generatePath = (points: { layer: number; prob: number }[]) => {
    if (points.length === 0) return '';
    const sorted = [...points].sort((a, b) => a.layer - b.layer);
    return sorted.map((p, i) => 
      `${i === 0 ? 'M' : 'L'} ${xScale(p.layer)} ${yScale(p.prob)}`
    ).join(' ');
  };

  const handleAddToken = () => {
    const trimmed = newTokenInput.trim();
    if (trimmed && !customTokens.includes(trimmed) && customTokens.length < 4) {
      setCustomTokens([...customTokens, trimmed]);
      setNewTokenInput('');
    }
  };

  const handleRemoveToken = (token: string) => {
    setCustomTokens(customTokens.filter(t => t !== token));
  };

  return (
    <div>
      {/* Position selector */}
      {data.positions.length > 1 && (
        <div style={{ marginBottom: '16px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {data.positions.map((pos, idx) => (
            <button
              key={idx}
              onClick={() => setSelectedPosition(idx)}
              className="font-mono"
              style={{
                padding: '6px 12px',
                border: selectedPosition === idx ? '2px solid #000' : '1px solid #ddd',
                borderRadius: '4px',
                background: selectedPosition === idx ? '#f0f0f0' : '#fff',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: selectedPosition === idx ? 600 : 400,
              }}
              title={windowTokens[idx]?.token || `Position ${pos.position}`}
            >
              {windowTokens[idx] 
                ? `"${safeTokenDisplay(windowTokens[idx].token, 8)}"` 
                : `Pos ${pos.position}`}
            </button>
          ))}
        </div>
      )}

      {/* Add custom token */}
      <div style={{ 
        marginBottom: '16px', 
        display: 'flex', 
        alignItems: 'center', 
        gap: '8px',
        padding: '12px',
        background: '#f8f8f8',
        borderRadius: '6px',
      }}>
        <span style={{ fontSize: '12px', color: '#666' }}>Track token:</span>
        <input
          type="text"
          value={newTokenInput}
          onChange={(e) => setNewTokenInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleAddToken()}
          placeholder="Enter token to track..."
          className="font-mono"
          style={{
            padding: '6px 10px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            fontSize: '12px',
            width: '150px',
          }}
        />
        <button
          onClick={handleAddToken}
          disabled={!newTokenInput.trim() || customTokens.length >= 4}
          style={{
            padding: '6px 12px',
            background: newTokenInput.trim() && customTokens.length < 4 ? '#000' : '#ccc',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            fontSize: '12px',
            cursor: newTokenInput.trim() && customTokens.length < 4 ? 'pointer' : 'default',
          }}
        >
          Add
        </button>
        {customTokens.length > 0 && (
          <span style={{ fontSize: '11px', color: '#999', marginLeft: '8px' }}>
            Custom: {customTokens.length}/4
          </span>
        )}
      </div>

      {/* Legend */}
      <div style={{ marginBottom: '12px', display: 'flex', flexWrap: 'wrap', gap: '12px', fontSize: '12px' }}>
        {series.map(s => (
          <div 
            key={s.token} 
            style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '6px',
              padding: '4px 8px',
              background: s.isCustom ? '#e8f4ff' : 'transparent',
              borderRadius: '4px',
            }}
          >
            <div style={{ width: '16px', height: '3px', background: s.color, borderRadius: '2px' }} />
            <span className="font-mono">"{safeTokenDisplay(s.token, 12)}"</span>
            {s.isCustom && (
              <button
                onClick={() => handleRemoveToken(s.token)}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '14px',
                  color: '#999',
                  padding: '0 2px',
                }}
              >
                ×
              </button>
            )}
          </div>
        ))}
      </div>

      {/* SVG Chart */}
      <div style={{ 
        background: '#fafafa', 
        borderRadius: '8px', 
        border: '1px solid #e5e5e5',
        padding: '16px',
        overflowX: 'auto',
        position: 'relative',
      }}>
        <svg width={chartWidth} height={chartHeight} style={{ display: 'block' }}>
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map(prob => (
            <g key={prob}>
              <line
                x1={padding.left}
                y1={yScale(prob)}
                x2={chartWidth - padding.right}
                y2={yScale(prob)}
                stroke="#e0e0e0"
                strokeDasharray={prob === 0 ? 'none' : '4,4'}
              />
              <text
                x={padding.left - 8}
                y={yScale(prob)}
                textAnchor="end"
                alignmentBaseline="middle"
                fontSize="10"
                fill="#999"
              >
                {(prob * 100).toFixed(0)}%
              </text>
            </g>
          ))}

          {/* X-axis labels */}
          {[0, Math.floor(numLayers / 4), Math.floor(numLayers / 2), Math.floor(3 * numLayers / 4), numLayers - 1].map(layer => (
            <text
              key={layer}
              x={xScale(layer)}
              y={chartHeight - padding.bottom + 20}
              textAnchor="middle"
              fontSize="10"
              fill="#999"
            >
              L{layer}
            </text>
          ))}

          {/* Axis labels */}
          <text
            x={chartWidth / 2}
            y={chartHeight - 5}
            textAnchor="middle"
            fontSize="11"
            fill="#666"
          >
            Layer
          </text>
          <text
            x={15}
            y={chartHeight / 2}
            textAnchor="middle"
            fontSize="11"
            fill="#666"
            transform={`rotate(-90, 15, ${chartHeight / 2})`}
          >
            Probability
          </text>

          {/* Lines */}
          {series.map(s => (
            <path
              key={s.token}
              d={generatePath(s.points)}
              fill="none"
              stroke={s.color}
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeDasharray={s.isCustom ? '6,3' : 'none'}
            />
          ))}

          {/* Data points */}
          {series.map(s => 
            s.points.map(p => (
              <circle
                key={`${s.token}-${p.layer}`}
                cx={xScale(p.layer)}
                cy={yScale(p.prob)}
                r="4"
                fill={s.color}
                stroke="#fff"
                strokeWidth="1.5"
                style={{ cursor: 'pointer' }}
                onMouseEnter={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  setHoveredPoint({ 
                    layer: p.layer, 
                    token: s.token, 
                    prob: p.prob,
                    x: rect.left + rect.width / 2,
                    y: rect.top,
                  });
                }}
                onMouseLeave={() => setHoveredPoint(null)}
              />
            ))
          )}
        </svg>

        {/* Hover tooltip */}
        {hoveredPoint && (
          <div style={{
            position: 'fixed',
            left: hoveredPoint.x,
            top: hoveredPoint.y - 8,
            transform: 'translate(-50%, -100%)',
            background: '#fff',
            border: '1px solid #ddd',
            borderRadius: '6px',
            padding: '10px 14px',
            boxShadow: '0 4px 16px rgba(0,0,0,0.2)',
            fontSize: '12px',
            zIndex: 10000,
            pointerEvents: 'none',
          }}>
            <div className="font-mono" style={{ fontWeight: 600 }}>
              "{safeTokenDisplay(hoveredPoint.token, 12)}"
            </div>
            <div style={{ color: '#666' }}>
              Layer {hoveredPoint.layer}: {(hoveredPoint.prob * 100).toFixed(2)}%
            </div>
          </div>
        )}
      </div>

      {/* Final layer summary */}
      <div style={{ 
        marginTop: '16px', 
        padding: '12px 16px',
        background: '#f0f0f0',
        borderRadius: '6px',
        fontSize: '13px',
      }}>
        <strong>Final prediction:</strong>{' '}
        <span className="font-mono">
          {position.layer_predictions[position.layer_predictions.length - 1]?.predictions[0]
            ? `"${safeTokenDisplay(position.layer_predictions[position.layer_predictions.length - 1].predictions[0].token)}" 
               (${(position.layer_predictions[position.layer_predictions.length - 1].predictions[0].probability * 100).toFixed(1)}%)`
            : 'N/A'}
        </span>
      </div>
    </div>
  );
}
