import React from 'react';
import styles from './PillNavbar.module.css';

export type NavItem = {
  id: string;
  label: string;
  icon?: React.ReactNode;
};

interface PillNavbarProps {
  items: NavItem[];
  activeId: string;
  onSelect: (id: string) => void;
  actionButton?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

export function PillNavbar({ items, activeId, onSelect, actionButton, className = '' }: PillNavbarProps) {
  return (
    <nav className={`${styles.outer} ${className}`} aria-label="Primary navigation">
      <div className={styles.capsule}>
        <div className={styles.track}>
          {items.map(item => (
            <button
              key={item.id}
              className={`${styles.pill} ${item.id === activeId ? styles.active : ''}`}
              onClick={() => onSelect(item.id)}
              aria-current={item.id === activeId ? 'page' : undefined}
            >
              {item.icon && <span className={styles.icon} aria-hidden="true">{item.icon}</span>}
              <span className={styles.label}>{item.label}</span>
            </button>
          ))}
          {actionButton && (
            <button
              className={styles.actionPill}
              onClick={actionButton.onClick}
            >
              <span className={styles.label}>{actionButton.label}</span>
            </button>
          )}
        </div>
      </div>
    </nav>
  );
}

