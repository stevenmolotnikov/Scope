import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Scope - Token Probability Visualizer',
  description: 'Chat with AI and see token probabilities in real-time',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
