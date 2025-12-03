'use client';

import { useEffect } from 'react';
import { useUIStore } from '@/stores/uiStore';

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const { theme, _hasHydrated } = useUIStore();

  useEffect(() => {
    if (!_hasHydrated) return;

    const root = document.documentElement;
    root.classList.remove('light', 'dark');
    
    if (theme !== 'system') {
      root.classList.add(theme);
    }
  }, [theme, _hasHydrated]);

  return <>{children}</>;
}

