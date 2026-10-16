import { useState, useCallback, useRef } from 'react';
import { Card } from '../common/Card';
import { PillButton } from '../common/PillButton';
import type { ExampleSnippet } from '../../api/types';
import styles from './CodeEditor.module.css';

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  loading: boolean;
  examples: ExampleSnippet[];
  // Generation settings
  numBeams: number;
  maxNewTokens: number;
  includeMermaid: boolean;
  onSettingsChange: (settings: Partial<{ numBeams: number; maxNewTokens: number; includeMermaid: boolean }>) => void;
}

export function CodeEditor({
  value, onChange, onSubmit, loading, examples,
  numBeams, maxNewTokens, includeMermaid, onSettingsChange,
}: CodeEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [showSettings, setShowSettings] = useState(false);
  const lineCount = value.split('\n').length;

  const handleExampleSelect = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const ex = examples.find(x => x.id === e.target.value);
    if (ex) onChange(ex.code);
    e.target.value = '';
  }, [examples, onChange]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const el = e.currentTarget;
      const start = el.selectionStart;
      const end = el.selectionEnd;
      const newVal = value.substring(0, start) + '    ' + value.substring(end);
      onChange(newVal);
      requestAnimationFrame(() => {
        el.selectionStart = el.selectionEnd = start + 4;
      });
    }
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!loading) onSubmit();
    }
  }, [value, onChange, loading, onSubmit]);

  return (
    <Card
      title="Python Source Code"
      headerActions={
        <div className={styles.headerActions}>
          {examples.length > 0 && (
            <select
              className={styles.exampleSelect}
              defaultValue=""
              onChange={handleExampleSelect}
              aria-label="Load example code"
            >
              <option value="" disabled>Load example…</option>
              {examples.map(ex => (
                <option key={ex.id} value={ex.id}>{ex.title}</option>
              ))}
            </select>
          )}
          <button
            className={styles.settingsToggle}
            onClick={() => setShowSettings(s => !s)}
            aria-expanded={showSettings}
            aria-label="Toggle generation settings"
            title="Generation settings"
          >
            ⚙
          </button>
          <PillButton
            size="sm"
            onClick={onSubmit}
            loading={loading}
            disabled={!value.trim()}
            title="Generate documentation (Ctrl+Enter)"
          >
            {loading ? 'Generating…' : 'Generate'}
          </PillButton>
        </div>
      }
    >
      {showSettings && (
        <div className={styles.settings}>
          <label className={styles.settingItem}>
            <span>Beams</span>
            <input
              type="range" min={1} max={5} step={1} value={numBeams}
              onChange={e => onSettingsChange({ numBeams: Number(e.target.value) })}
            />
            <span className={styles.settingVal}>{numBeams}</span>
          </label>
          <label className={styles.settingItem}>
            <span>Max tokens</span>
            <input
              type="range" min={16} max={512} step={16} value={maxNewTokens}
              onChange={e => onSettingsChange({ maxNewTokens: Number(e.target.value) })}
            />
            <span className={styles.settingVal}>{maxNewTokens}</span>
          </label>
          <label className={styles.settingCheckbox}>
            <input
              type="checkbox" checked={includeMermaid}
              onChange={e => onSettingsChange({ includeMermaid: e.target.checked })}
            />
            Include Mermaid flowchart
          </label>
        </div>
      )}
      <div className={styles.editorWrap}>
        <div className={styles.lineNumbers} aria-hidden="true">
          {Array.from({ length: Math.max(lineCount, 1) }, (_, i) => (
            <span key={i + 1}>{i + 1}</span>
          ))}
        </div>
        <textarea
          ref={textareaRef}
          className={styles.textarea}
          value={value}
          onChange={e => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          spellCheck={false}
          autoCorrect="off"
          autoCapitalize="off"
          placeholder={"def greet(name: str) -> str:\n    ...\n\n# Press Ctrl+Enter to generate docs"}
          aria-label="Python source code input"
        />
      </div>
      <div className={styles.footer}>
        <span className={styles.hint}>Ctrl+Enter to generate • Tab for indent</span>
        <span className={styles.charCount}>{value.length.toLocaleString()} chars</span>
      </div>
    </Card>
  );
}
