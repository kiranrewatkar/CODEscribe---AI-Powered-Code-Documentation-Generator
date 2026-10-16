import type { ExampleSnippet } from '../../api/types';
import { Card } from '../common/Card';
import styles from './ExamplesPanel.module.css';

interface ExamplesPanelProps {
  examples: ExampleSnippet[];
  loading: boolean;
  onSelect: (code: string) => void;
}

export function ExamplesPanel({ examples, loading, onSelect }: ExamplesPanelProps) {
  if (loading) {
    return (
      <Card title="Examples">
        <p className={styles.hint}>Loading examples…</p>
      </Card>
    );
  }

  if (!examples.length) {
    return (
      <Card title="Examples">
        <p className={styles.hint}>No examples available.</p>
      </Card>
    );
  }

  return (
    <Card title="Code Examples">
      <div className={styles.list}>
        {examples.map(ex => (
          <div key={ex.id} className={styles.item}>
            <div className={styles.itemHeader}>
              <span className={styles.itemTitle}>{ex.title}</span>
            </div>
            <p className={styles.itemDesc}>{ex.description}</p>
            <button
              className={styles.loadBtn}
              onClick={() => onSelect(ex.code)}
              aria-label={`Load example: ${ex.title}`}
            >
              Load into editor →
            </button>
          </div>
        ))}
      </div>
    </Card>
  );
}
