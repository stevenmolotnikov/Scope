'use client';

import { Token } from './Token';
import { useUIStore } from '@/stores/uiStore';
import { cleanTokenizerArtifacts } from '@/lib/utils';
import { cn } from '@/lib/utils';
import type { Token as TokenType } from '@/types';
import ReactMarkdown from 'react-markdown';

interface TokenizedMessageProps {
  tokens: TokenType[];
  messageId: string;
}

export function TokenizedMessage({ tokens, messageId }: TokenizedMessageProps) {
  const { viewMode } = useUIStore();

  if (!tokens || tokens.length === 0) {
    return (
      <span className="text-muted-foreground italic">Generating...</span>
    );
  }

  // Markdown Text View
  if (viewMode === 'text') {
    const fullText = tokens.map(t => cleanTokenizerArtifacts(t.token)).join('');
    return (
      <div className="prose prose-sm dark:prose-invert max-w-none leading-relaxed">
        <ReactMarkdown>{fullText}</ReactMarkdown>
      </div>
    );
  }

  // Token View (Default & Diff)
  return (
    <div className="font-mono text-sm leading-relaxed">
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
