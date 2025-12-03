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
    <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground px-2.5 py-2 border-b border-border bg-muted font-mono">
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
    <div className="border-b border-border">
      {alternatives.slice(0, 3).map((alt, idx) => {
        const actualIndex = idx + indexOffset;
        const isSelected = selectedIndex === actualIndex;
        return (
          <div
            key={idx}
            onClick={() => onSelect(alt, actualIndex)}
            className={`font-mono flex items-center justify-between px-2.5 py-1.5 text-[11px] cursor-pointer transition-colors ${
              isSelected ? 'bg-accent' : 'bg-background hover:bg-muted'
            } ${idx < 2 ? 'border-b border-border/50' : ''}`}
          >
            <div className="flex items-center gap-1.5">
              <span className="text-muted-foreground text-[9px] w-4">#{alt.rank}</span>
              <span className={isSelected ? 'font-semibold' : ''}>"{formatTokenForDisplay(alt.token)}"</span>
            </div>
            <span className="text-muted-foreground">{formatProbability(alt.probability)}</span>
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
    <div className="w-[300px] shrink-0 flex flex-col bg-background border-l border-border h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground m-0">
          Inspector
        </h3>
        <button
          onClick={toggleRightSidebar}
          className="p-1 bg-transparent border-none cursor-pointer text-muted-foreground rounded hover:text-foreground hover:bg-muted transition-colors"
        >
          <X size={16} />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        {!token ? (
          <div className="flex flex-col items-center justify-center h-[200px] text-muted-foreground text-center">
            <p className="text-[13px]">Click any token to inspect</p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            
            {/* ==================== TOKEN BOX ==================== */}
            <div className="bg-muted border border-border rounded-lg p-4 text-center">
              {/* Token Display */}
              <div className="font-mono text-[22px] font-semibold break-all mb-3">
                "{formatTokenForDisplay(token.token)}"
              </div>

              {/* Standard View: Show Prob + Rank */}
              {!isDiffView && (
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <div className="text-muted-foreground text-[10px] mb-0.5">Probability</div>
                    <div className="font-mono font-semibold">
                      {formatProbability(token.probability)}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground text-[10px] mb-0.5">Rank</div>
                    <div className="font-mono font-semibold">
                      #{token.rank ?? 'N/A'}
                    </div>
                  </div>
                </div>
              )}

              {/* Diff View: Show Diff Badge */}
              {isDiffView && token.diff_data && (
                <div className={`px-4 py-2.5 rounded-md flex justify-between items-center ${
                  token.diff_data.prob_diff > 0 
                    ? 'bg-green-100 border border-green-300 dark:bg-green-950 dark:border-green-800' 
                    : token.diff_data.prob_diff < 0 
                    ? 'bg-red-100 border border-red-300 dark:bg-red-950 dark:border-red-800' 
                    : 'bg-muted border border-border'
                }`}>
                  <span className="text-xs text-muted-foreground">Diff</span>
                  <span className={`font-mono text-sm font-semibold ${
                    token.diff_data.prob_diff > 0 ? 'text-green-600 dark:text-green-400' : token.diff_data.prob_diff < 0 ? 'text-red-600 dark:text-red-400' : 'text-muted-foreground'
                  }`}>
                    {token.diff_data.prob_diff > 0 ? '+' : ''}{token.diff_data.prob_diff.toFixed(1)}%
                  </span>
                </div>
              )}
            </div>

            {/* ==================== STANDARD VIEW: MODEL BOX ==================== */}
            {!isDiffView && (
              <div className="border border-border rounded-lg overflow-hidden">
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
                <div className="p-2">
                  <button
                    onClick={() => handleLogitLens()}
                    className="w-full p-2 bg-muted border border-border rounded text-[11px] font-medium cursor-pointer text-muted-foreground flex items-center justify-center gap-1.5 hover:bg-accent hover:text-accent-foreground transition-colors"
                  >
                    <Search size={12} /> LogitLens
                  </button>
                </div>
              </div>
            )}

            {/* ==================== DIFF VIEW: GENERATION BOX ==================== */}
            {isDiffView && (
              <div className="border border-border rounded-lg overflow-hidden">
                {/* Header */}
                <SectionHeader>
                  Generation · {formatModelName(conversation?.model || '')}
                </SectionHeader>

                {/* Stats */}
                <div className="grid grid-cols-2 gap-2 p-2.5 border-b border-border text-[11px]">
                  <div>
                    <div className="text-muted-foreground text-[9px] mb-0.5">Probability</div>
                    <div className="font-mono font-semibold">
                      {formatProbability(token.probability)}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground text-[9px] mb-0.5">Rank</div>
                    <div className="font-mono font-semibold">
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
                <div className="p-2">
                  <button
                    onClick={() => handleLogitLens(conversation?.model)}
                    className="w-full p-2 bg-muted border border-border rounded text-[11px] font-medium cursor-pointer text-muted-foreground flex items-center justify-center gap-1.5 hover:bg-accent hover:text-accent-foreground transition-colors"
                  >
                    <Search size={12} /> LogitLens
                  </button>
                </div>
              </div>
            )}

            {/* ==================== DIFF VIEW: ANALYSIS BOX ==================== */}
            {isDiffView && token.diff_data && (
              <div className="border border-border rounded-lg overflow-hidden">
                {/* Header */}
                <SectionHeader>
                  Analysis · {formatModelName(token.diff_data.analysis_model || '')}
                </SectionHeader>

                {/* Stats */}
                <div className="grid grid-cols-2 gap-2 p-2.5 border-b border-border text-[11px]">
                  <div>
                    <div className="text-muted-foreground text-[9px] mb-0.5">Probability</div>
                    <div className="font-mono font-semibold">
                      {formatProbability(token.diff_data.analysis_prob)}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground text-[9px] mb-0.5">Rank</div>
                    <div className="font-mono font-semibold">
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
                <div className="p-2">
                  <button
                    onClick={() => handleLogitLens(token.diff_data?.analysis_model)}
                    className="w-full p-2 bg-muted border border-border rounded text-[11px] font-medium cursor-pointer text-muted-foreground flex items-center justify-center gap-1.5 hover:bg-accent hover:text-accent-foreground transition-colors"
                  >
                    <Search size={12} /> LogitLens
                  </button>
                </div>
              </div>
            )}

            {/* ==================== RULE APPLIED ==================== */}
            {token.rule_applied && (
              <div className="bg-yellow-50 dark:bg-yellow-950 border border-yellow-300 dark:border-yellow-800 rounded-lg p-2.5 text-[11px]">
                <div className="font-semibold mb-1">
                  ⚙ {token.rule_applied.name}
                </div>
                <div className="text-muted-foreground">{token.rule_applied.reason}</div>
              </div>
            )}

            {/* ==================== INJECT ==================== */}
            <div className="border border-border rounded-lg overflow-hidden">
              <SectionHeader>Inject</SectionHeader>
              <div className="p-2.5 flex gap-1.5">
                <input
                  type="text"
                  value={injectionText}
                  onChange={(e) => {
                    setInjectionText(e.target.value);
                    setSelectedAltIndex(null);
                  }}
                  placeholder="Text to inject..."
                  className="font-mono flex-1 px-2.5 py-2 border border-input rounded text-xs bg-background focus:outline-none focus:ring-1 focus:ring-ring"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleInject();
                  }}
                />
                <button
                  onClick={handleInject}
                  disabled={!injectionText.trim()}
                  className={`px-3 py-2 border-none rounded text-xs font-medium ${
                    injectionText.trim() 
                      ? 'bg-primary text-primary-foreground cursor-pointer hover:bg-primary/90' 
                      : 'bg-muted text-muted-foreground cursor-default'
                  }`}
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
