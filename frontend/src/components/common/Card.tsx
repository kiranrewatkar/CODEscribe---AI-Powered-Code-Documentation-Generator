import React from 'react';
import styles from './Card.module.css';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  headerActions?: React.ReactNode;
  noPadding?: boolean;
}

export function Card({ children, className = '', title, headerActions, noPadding }: CardProps) {
  return (
    <div className={`${styles.card} ${className}`}>
      {(title || headerActions) && (
        <div className={styles.header}>
          {title && <h3 className={styles.title}>{title}</h3>}
          {headerActions && <div className={styles.actions}>{headerActions}</div>}
        </div>
      )}
      <div className={noPadding ? styles.bodyNoPad : styles.body}>
        {children}
      </div>
    </div>
  );
}
