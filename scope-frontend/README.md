# Scope Frontend

A Next.js TypeScript frontend for the Scope token probability visualizer.

## Features

- Real-time streaming chat with token-level probability visualization
- Token highlighting by probability or rank
- DiffLens - cross-model probability comparison
- Logit Lens - layer-by-layer analysis with heatmap and chart views
- Token Inspector sidebar with injection capabilities
- Conversation management with tree-based message navigation
- Automation rules for generation control
- System prompts and sampling settings

## Tech Stack

- **Next.js 15** with App Router
- **TypeScript** for type safety
- **TailwindCSS v4** for styling
- **Zustand** for state management
- **Chart.js** for visualizations

## Getting Started

### Prerequisites

- Node.js 18+
- Flask backend running on port 5001

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

The app will be available at [http://localhost:3000](http://localhost:3000).

### Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:5001
```

## Project Structure

```
src/
├── app/                    # Next.js App Router
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Main chat page
│   └── globals.css        # Global styles
├── components/
│   ├── chat/              # Chat components
│   ├── inspector/         # Token inspector sidebar
│   ├── sidebar/           # Conversations sidebar
│   ├── modals/            # Modal dialogs
│   └── ui/                # Reusable UI primitives
├── hooks/                 # Custom React hooks
├── lib/                   # API client, utilities
├── stores/                # Zustand state stores
└── types/                 # TypeScript type definitions
```

## Backend Integration

The frontend communicates with the Flask backend via:

- **REST API** (`/api/*`) for conversations, token search, analysis
- **SSE** (`/stream`) for real-time token generation

Make sure the Flask backend has CORS enabled for `localhost:3000`.
