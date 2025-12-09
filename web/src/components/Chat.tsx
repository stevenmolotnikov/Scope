import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { useState } from 'react';
import { cn } from '@/lib/utils';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8787';

export function Chat() {
  const [input, setInput] = useState('');
  const { messages, sendMessage, status } = useChat({
    transport: new DefaultChatTransport({
      api: `${API_URL}/api/chat`,
    }),
  });

  const isLoading = status === 'streaming' || status === 'submitted';

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      sendMessage({ text: input });
      setInput('');
    }
  };

  const getTextContent = (parts: typeof messages[0]['parts']) => {
    return parts
      .filter((part): part is Extract<typeof part, { type: 'text' }> => part.type === 'text')
      .map((part) => part.text)
      .join('');
  };

  return (
    <div className="flex flex-col max-w-3xl mx-auto h-[calc(100vh-200px)] border border-border rounded-lg overflow-hidden bg-card">
      <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            <p>Start a conversation by typing a message below.</p>
          </div>
        )}
        {messages.map((message) => (
          <div
            key={message.id}
            className={cn(
              'flex flex-col gap-1 p-3 rounded-lg max-w-[80%]',
              message.role === 'user'
                ? 'self-end bg-primary text-primary-foreground'
                : 'self-start bg-muted text-muted-foreground'
            )}
          >
            <div className="text-xs font-semibold opacity-70 uppercase tracking-wide">
              {message.role === 'user' ? 'You' : 'Assistant'}
            </div>
            <div className="whitespace-pre-wrap break-words leading-relaxed">
              {getTextContent(message.parts)}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex flex-col gap-1 p-3 rounded-lg max-w-[80%] self-start bg-muted text-muted-foreground">
            <div className="text-xs font-semibold opacity-70 uppercase tracking-wide">Assistant</div>
            <div className="whitespace-pre-wrap break-words leading-relaxed">Thinking...</div>
          </div>
        )}
      </div>
      <form onSubmit={handleSubmit} className="flex gap-2 p-4 border-t border-border bg-card">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={isLoading}
          className="flex-1 px-4 py-2 border border-input rounded-md bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="px-6 py-2 bg-primary text-primary-foreground rounded-md font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity whitespace-nowrap"
        >
          Send
        </button>
      </form>
    </div>
  );
}
