'use client';

import { forwardRef, type SelectHTMLAttributes } from 'react';
import { cn } from '@/lib/utils';

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  variant?: 'default' | 'minimal';
}

const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, variant = 'default', children, ...props }, ref) => {
    return (
      <select
        ref={ref}
        className={cn(
          'appearance-none bg-no-repeat cursor-pointer transition-all',
          'bg-[url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20width%3D%2210%22%20height%3D%226%22%20viewBox%3D%220%200%2010%206%22%20fill%3D%22none%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%3Cpath%20d%3D%22M1%201L5%205L9%201%22%20stroke%3D%22%23666666%22%20stroke-width%3D%221.5%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22/%3E%3C/svg%3E")]',
          'bg-[length:8px] bg-[right_10px_center]',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          // Variants
          variant === 'default' &&
            'px-3 py-2 pr-8 border border-gray-200 rounded-lg bg-white text-sm font-mono hover:border-gray-400 focus:outline-none focus:border-black',
          variant === 'minimal' &&
            'px-3 py-1.5 pr-7 border border-transparent rounded-md bg-transparent text-sm font-mono text-gray-500 font-medium hover:bg-gray-200 hover:text-black focus:outline-none focus:bg-gray-100 focus:border-gray-400',
          className
        )}
        {...props}
      >
        {children}
      </select>
    );
  }
);

Select.displayName = 'Select';

export { Select };

