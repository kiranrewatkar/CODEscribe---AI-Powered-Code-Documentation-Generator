import { useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import { Card } from '../common/Card';
import { PillButton } from '../common/PillButton';
import { LoadingSpinner } from '../common/LoadingSpinner';
import type { DocumentResponse } from '../../api/types';
import styles from './DocViewer.module.css';

interface DocViewerProps {
  result: DocumentResponse | null;
  loading: boolean;
  error: string | null;
  onClearError: () => void;
}

export function DocViewer({ result, loading, error, onClearError }: DocViewerProps) {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  const [activeTab, setActiveTab] = useState<'rendered' | 'raw'>('rendered');

  const handleCopy = useCallback(async () => {
    if (!result?.markdown) return;
    try {
      await navigator.clipboard.writeText(result.markdown);
      setCopied(true);
      setCopyError(false);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopyError(true);
      setTimeout(() => setCopyError(false), 2500);
    }
  }, [result?.markdown]);

  const headerActions = result?.markdown ? (
    <div className={styles.tabsAndCopy}>
      <div className={styles.tabs}>
        <button
          className={`${styles.tab} ${activeTab === 'rendered' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('rendered')}
        >
          Preview
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'raw' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('raw')}
        >
          Markdown
        </button>
      </div>
      <PillButton
        size="sm"
        variant={copied ? 'primary' : copyError ? 'danger' : 'secondary'}
        onClick={handleCopy}
      >
        {copied ? '✓ Copied' : copyError ? '✕ Failed' : '⎘ Copy'}
      </PillButton>
    </div>
  ) : undefined;

  return (
    <Card title="Documentation Output" headerActions={headerActions} noPadding={!!(result || loading || error)}>
      <div className={styles.content}>
        {loading && (
          <div className={styles.centered}>
            <LoadingSpinner size="lg" label="Generating documentation…" />
          </div>
        )}

        {!loading && error && (
          <div className={styles.errorState}>
            <div className={styles.errorIcon}>⚠</div>
            <p className={styles.errorText}>{error}</p>
            <PillButton size="sm" variant="secondary" onClick={onClearError}>
              Clear
            </PillButton>
          </div>
        )}

        {!loading && !error && !result && (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>📄</div>
            <p className={styles.emptyTitle}>No documentation yet</p>
            <p className={styles.emptyHint}>
              Paste Python code in the editor and click <strong>Generate</strong>.
            </p>
          </div>
        )}

        {!loading && !error && result && (
          <div className={styles.resultWrap}>
            {activeTab === 'rendered' ? (
              <div className={styles.markdown}>
                <ReactMarkdown>{result.markdown}</ReactMarkdown>
              </div>
            ) : (
              <pre className={styles.rawMarkdown}>{result.markdown}</pre>
            )}

            {result.timing_ms && (
              <div className={styles.timingBar}>
                <span>Inference: {(result.timing_ms.inference_ms / 1000).toFixed(2)}s</span>
                <span>Total: {(result.timing_ms.total_ms / 1000).toFixed(2)}s</span>
                <span>Device: {result.device}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}
