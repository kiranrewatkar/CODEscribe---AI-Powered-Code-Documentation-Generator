import React from 'react';
import styles from './Sidebar.module.css';

export interface SidebarItem {
  id: string;
  label: string;
  icon: React.ReactNode;
}

interface SidebarProps {
  items: SidebarItem[];
  activeId: string;
  onSelect: (id: string) => void;
}

export function Sidebar({ items, activeId, onSelect }: SidebarProps) {
  return (
    <aside className={styles.sidebar} aria-label="Workspace sidebar">
      <div className={styles.logo}>
        <span className={styles.logoMark}>CS</span>
        <span className={styles.logoText}>CODEscribe</span>
      </div>
      <nav className={styles.nav}>
        {items.map(item => (
          <button
            key={item.id}
            className={`${styles.item} ${item.id === activeId ? styles.active : ''}`}
            onClick={() => onSelect(item.id)}
            aria-current={item.id === activeId ? 'page' : undefined}
          >
            <span className={styles.itemIcon} aria-hidden="true">{item.icon}</span>
            <span className={styles.itemLabel}>{item.label}</span>
          </button>
        ))}
      </nav>
    </aside>
  );
}
