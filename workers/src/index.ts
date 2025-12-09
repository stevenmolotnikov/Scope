import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { streamText, convertToModelMessages } from 'ai';
import { createOpenRouter } from '@openrouter/ai-sdk-provider';

type Env = {
  OPENROUTER_API_KEY: string;
};

const app = new Hono<{ Bindings: Env }>();

app.use('/*', cors({
  origin: '*',
  allowMethods: ['GET', 'POST', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
}));

app.post('/api/chat', async (c) => {
  const { messages } = await c.req.json();

  if (!messages || !Array.isArray(messages)) {
    return c.json({ error: 'Messages array is required' }, 400);
  }

  const openrouter = createOpenRouter({
    apiKey: c.env.OPENROUTER_API_KEY,
  });

  const result = streamText({
    model: openrouter.chat('openai/gpt-4o-mini'),
    messages: convertToModelMessages(messages),
  });

  return result.toUIMessageStreamResponse();
});

app.get('/health', (c) => {
  return c.json({ status: 'ok' });
});

export default app;
