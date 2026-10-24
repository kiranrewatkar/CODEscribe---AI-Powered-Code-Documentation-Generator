import React, { useState, useCallback } from 'react';
import styles from './PillButton.module.css';

interface PillButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
  children: React.ReactNode;
  enableRipple?: boolean;
}

interface Ripple {
  x: number;
  y: number;
  size: number;
  id: number;
}

export function PillButton({
  variant = 'primary',
  size = 'md',
  loading = false,
  icon,
  children,
  disabled,
  enableRipple = true,
  className = '',
  onClick,
  ...rest
}: PillButtonProps) {
  const [ripples, setRipples] = useState<Ripple[]>([]);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLButtonElement>) => {
      if (disabled || loading) return;

      // Only add ripple if user doesn't prefer reduced motion
      const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (enableRipple && !prefersReducedMotion) {
        const btn = e.currentTarget;
        const rect = btn.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const rippleSize = Math.max(rect.width, rect.height) * 1.5;
        const newRipple: Ripple = { x, y, size: rippleSize, id: Date.now() };

        // Keep maximum of 1 active ripple to prevent DOM clutter
        setRipples([newRipple]);

        setTimeout(() => {
          setRipples((prev) => prev.filter((r) => r.id !== newRipple.id));
        }, 450);
      }

      if (onClick) {
        onClick(e);
      }
    },
    [disabled, loading, enableRipple, onClick]
  );

  return (
    <button
      {...rest}
      disabled={disabled || loading}
      onClick={handleClick}
      className={`${styles.btn} ${styles[variant]} ${styles[size]} ${loading ? styles.loading : ''} ${className}`}
    >
      {loading ? (
        <span className={styles.spinner} aria-hidden="true" />
      ) : icon ? (
        <span className={styles.icon} aria-hidden="true">{icon}</span>
      ) : null}
      <span className={styles.btnContent}>{children}</span>

      {/* Click Origin Ripple Layer */}
      {ripples.map((r) => (
        <span
          key={r.id}
          className={styles.ripple}
          style={{
            left: r.x,
            top: r.y,
            width: r.size,
            height: r.size,
          }}
          aria-hidden="true"
        />
      ))}
    </button>
  );
}
