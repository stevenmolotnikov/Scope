'use client';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'default' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  children: React.ReactNode;
}

export function Button({
  variant = 'default',
  size = 'md',
  children,
  disabled,
  style,
  ...props
}: ButtonProps) {
  const baseStyles: React.CSSProperties = {
    borderRadius: '6px',
    fontWeight: 500,
    cursor: disabled ? 'default' : 'pointer',
    border: 'none',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '6px',
    transition: 'all 0.1s ease',
    opacity: disabled ? 0.5 : 1,
  };

  const sizeStyles: Record<string, React.CSSProperties> = {
    sm: { padding: '6px 12px', fontSize: '12px' },
    md: { padding: '8px 16px', fontSize: '14px' },
    lg: { padding: '12px 24px', fontSize: '16px' },
  };

  const variantStyles: Record<string, React.CSSProperties> = {
    primary: {
      background: '#000',
      color: '#fff',
    },
    secondary: {
      background: '#f5f5f5',
      color: '#333',
      border: '1px solid #e5e5e5',
    },
    default: {
      background: '#fff',
      color: '#333',
      border: '1px solid #ddd',
    },
    danger: {
      background: '#dc2626',
      color: '#fff',
    },
  };

  return (
    <button
      style={{
        ...baseStyles,
        ...sizeStyles[size],
        ...variantStyles[variant],
        ...style,
      }}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  );
}
