'use client';

import { Token } from './Token';
import { useUIStore } from '@/stores/uiStore';
import { cn } from '@/lib/utils';
import type { Token as TokenType } from '@/types';

interface TokenizedMessageProps {
  tokens: TokenType[];
  messageId: string;
}

export function TokenizedMessage({ tokens, messageId }: TokenizedMessageProps) {
  const { viewMode } = useUIStore();

  if (!tokens || tokens.length === 0) {
    return (
      <span className="text-gray-400 italic">Generating...</span>
    );
  }

  return (
    <div
      className={cn(
        'font-mono text-sm leading-relaxed',
        viewMode === 'text' && 'whitespace-pre-wrap'
      )}
    >
      {tokens.map((token, index) => (
        <Token
          key={`${messageId}-${index}`}
          token={token}
          messageId={messageId}
          tokenIndex={index}
        />
      ))}
    </div>
  );
}

