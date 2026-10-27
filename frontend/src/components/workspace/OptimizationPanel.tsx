import { Card } from '../common/Card';
import { LoadingSpinner } from '../common/LoadingSpinner';
import type { OptimizeResponse } from '../../api/types';
import styles from './OptimizationPanel.module.css';

interface OptimizationPanelProps {
  result: OptimizeResponse | null;
  loading: boolean;
  error: string | null;
  onClose?: () => void;
}

export function OptimizationPanel({ result, loading, error, onClose }: OptimizationPanelProps) {
  if (!loading && !result && !error) {
    return null;
  }

  return (
    <Card
      title="Code Optimization Suggestions"
      headerActions={
        onClose ? (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close optimization panel"
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-muted, #888)',
              cursor: 'pointer',
              fontSize: '1rem',
              padding: '2px 6px',
            }}
          >
            ✕
          </button>
        ) : undefined
      }
    >
      <div className={styles.container}>
        {loading && (
          <div style={{ padding: '1.5rem', textAlign: 'center' }}>
            <LoadingSpinner size="md" label="Running AST optimization analysis…" />
          </div>
        )}

        {!loading && error && (
          <div className={styles.errorBanner} role="alert">
            <strong>Analysis Error:</strong> {error}
          </div>
        )}

        {!loading && result && !result.success && (
          <div className={styles.errorBanner} role="alert">
            <strong>Static Analysis Issue:</strong> {result.error_message ?? 'Failed to parse source code.'}
          </div>
        )}

        {!loading && result && result.success && (
          <>
            <div className={styles.header}>
              <div className={styles.titleGroup}>
                <span>Status:</span>
                {result.total_findings === 0 ? (
                  <span className={`${styles.badge} ${styles.badgeSuccess}`}>
                    ✓ Clean (0 findings)
                  </span>
                ) : (
                  <span className={styles.badge}>
                    {result.total_findings} Suggestion{result.total_findings > 1 ? 's' : ''}
                  </span>
                )}
              </div>
            </div>

            {result.total_findings === 0 ? (
              <div className={styles.cleanState}>
                No structural anti-patterns detected. Source code conforms to CODEscribe's inspected optimization rules (OPT001–OPT004).
              </div>
            ) : (
              <div className={styles.findingsList}>
                {result.findings.map((f, idx) => (
                  <div key={`${f.rule_id}-${f.line_number}-${f.column_offset}-${idx}`} className={styles.findingCard}>
                    <div className={styles.findingHeader}>
                      <span className={styles.ruleTag}>{f.rule_id}</span>
                      <span className={styles.findingTitle}>{f.title}</span>
                      <span className={styles.location}>
                        Line {f.line_number}:{f.column_offset}
                        {f.function_scope ? ` (${f.function_scope})` : ''}
                      </span>
                    </div>
                    <p className={styles.description}>{f.description}</p>
                    <div className={styles.suggestionBox}>
                      <span className={styles.suggestionLabel}>Suggestion:</span>
                      {f.suggestion}
                    </div>
                  </div>
                ))}
              </div>
            )}

            <p className={styles.disclaimer}>{result.disclaimer}</p>
          </>
        )}
      </div>
    </Card>
  );
}
