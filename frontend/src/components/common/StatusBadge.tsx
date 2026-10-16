import styles from './StatusBadge.module.css';

interface StatusBadgeProps {
  status: 'ok' | 'degraded' | 'loading' | 'error';
  label?: string;
}

const STATUS_LABELS: Record<string, string> = {
  ok: 'Model Ready',
  degraded: 'Model Not Loaded',
  loading: 'Connecting…',
  error: 'Backend Offline',
};

export function StatusBadge({ status, label }: StatusBadgeProps) {
  const text = label ?? STATUS_LABELS[status] ?? status;
  return (
    <span className={`${styles.badge} ${styles[status]}`} title={text}>
      <span className={styles.dot} />
      {text}
    </span>
  );
}
