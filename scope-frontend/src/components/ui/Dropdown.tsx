'use client';

import { useState, useRef, useEffect, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

interface DropdownProps {
  trigger: ReactNode;
  children: ReactNode;
  align?: 'left' | 'right';
  className?: string;
}

export function Dropdown({
  trigger,
  children,
  align = 'left',
  className,
}: DropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div ref={dropdownRef} className="relative">
      <div onClick={() => setIsOpen(!isOpen)}>{trigger}</div>

      {isOpen && (
        <div
          className={cn(
            'absolute top-full mt-1 min-w-[200px] max-w-[320px] bg-white border-2 border-gray-300 rounded-lg shadow-lg py-2 z-50',
            align === 'left' && 'left-0',
            align === 'right' && 'right-0',
            className
          )}
        >
          {children}
        </div>
      )}
    </div>
  );
}

interface DropdownItemProps {
  children: ReactNode;
  onClick?: () => void;
  className?: string;
}

export function DropdownItem({ children, onClick, className }: DropdownItemProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full px-4 py-2 text-left text-sm flex items-center gap-2 hover:bg-gray-100 transition-colors',
        className
      )}
    >
      {children}
    </button>
  );
}

interface DropdownGroupProps {
  label?: string;
  children: ReactNode;
  className?: string;
}

export function DropdownGroup({ label, children, className }: DropdownGroupProps) {
  return (
    <div className={cn('px-2 py-2', className)}>
      {label && (
        <span className="block px-2 pb-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
          {label}
        </span>
      )}
      {children}
    </div>
  );
}

export function DropdownDivider() {
  return <div className="h-px bg-gray-200 my-1" />;
}

