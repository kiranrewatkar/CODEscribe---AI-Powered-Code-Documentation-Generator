import { useState, useCallback, useRef } from 'react';
import { Card } from '../common/Card';
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
  // Optimization analysis
  onOptimize?: () => void;
  optimizing?: boolean;
}

export function CodeEditor({
  value, onChange, onSubmit, loading, examples,
  numBeams, maxNewTokens, includeMermaid, onSettingsChange,
  onOptimize, optimizing = false,
}: CodeEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCounterRef = useRef(0);
  const [showSettings, setShowSettings] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [fileMessage, setFileMessage] = useState<{ type: 'error' | 'success'; text: string } | null>(null);
  const lineCount = value.split('\n').length;

  const handleExampleSelect = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const ex = examples.find(x => x.id === e.target.value);
    if (ex) {
      onChange(ex.code);
      setFileMessage(null);
    }
    e.target.value = '';
  }, [examples, onChange]);

  const processSelectedFile = useCallback((file: File) => {
    const fileName = file.name || '';
    const isPython = fileName.toLowerCase().endsWith('.py');

    if (!isPython) {
      setFileMessage({
        type: 'error',
        text: `Cannot load "${fileName}": Only Python (.py) files are supported. Existing code was preserved.`,
      });
      return;
    }

    const reader = new FileReader();

    reader.onload = (event) => {
      try {
        const content = event.target?.result;
        if (typeof content !== 'string') {
          setFileMessage({
            type: 'error',
            text: `Failed to read "${fileName}": File content is not readable as text.`,
          });
          return;
        }

        if (!content.trim()) {
          setFileMessage({
            type: 'error',
            text: `File "${fileName}" is empty. Please select a Python file with source code.`,
          });
          return;
        }

        onChange(content);
        setFileMessage({
          type: 'success',
          text: `Loaded "${fileName}" (${content.length.toLocaleString()} characters).`,
        });
      } catch (err) {
        setFileMessage({
          type: 'error',
          text: `An error occurred while reading "${fileName}". Existing code was preserved.`,
        });
      }
    };

    reader.onerror = () => {
      setFileMessage({
        type: 'error',
        text: `Failed to read "${fileName}". Existing code was preserved.`,
      });
    };

    reader.readAsText(file);
  }, [onChange]);

  const handleDragEnter = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current += 1;
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = 'copy';
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current -= 1;
    if (dragCounterRef.current <= 0) {
      dragCounterRef.current = 0;
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current = 0;
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (!files || files.length === 0) {
      return;
    }

    if (files.length > 1) {
      setFileMessage({
        type: 'error',
        text: 'Please drop only a single Python (.py) file at a time.',
      });
      return;
    }

    processSelectedFile(files[0]);
  }, [processSelectedFile]);

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) {
      return;
    }

    if (files.length > 1) {
      setFileMessage({
        type: 'error',
        text: 'Please select only a single Python (.py) file at a time.',
      });
      e.target.value = '';
      return;
    }

    processSelectedFile(files[0]);
    e.target.value = '';
  }, [processSelectedFile]);

  const handleTriggerFileInput = useCallback(() => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }, []);

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
          <input
            ref={fileInputRef}
            type="file"
            accept=".py"
            onChange={handleFileInputChange}
            className={styles.hiddenFileInput}
            tabIndex={-1}
            aria-hidden="true"
          />
          <button
            type="button"
            className={styles.importBtn}
            onClick={handleTriggerFileInput}
            title="Import a Python (.py) file from your computer"
          >
            <span>📂</span>
            <span>Import</span>
          </button>
          <button
            className={styles.settingsToggle}
            onClick={() => setShowSettings(s => !s)}
            aria-expanded={showSettings}
            aria-label="Toggle generation settings"
            title="Generation settings"
          >
            ⚙
          </button>
          <div className={styles.btnGroup}>
            {onOptimize && (
              <button
                type="button"
                className={styles.optimizeBtn}
                onClick={onOptimize}
                disabled={!value.trim() || loading || optimizing}
                title="Analyze Python code for static optimization opportunities"
              >
                <span>⚡</span>
                <span>{optimizing ? 'Analyzing…' : 'Optimize'}</span>
              </button>
            )}
            <button
              type="button"
              className={styles.generateBtn}
              onClick={onSubmit}
              disabled={!value.trim() || loading || optimizing}
              title="Generate documentation (Ctrl+Enter)"
            >
              <span>✨</span>
              <span>{loading ? 'Generating…' : 'Generate Documentation'}</span>
            </button>
          </div>
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
      <div
        className={`${styles.editorWrap} ${isDragging ? styles.editorWrapDragging : ''}`}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isDragging && (
          <div className={styles.dragOverlay} aria-hidden="true">
            <span className={styles.dragOverlayIcon}>📄</span>
            <span className={styles.dragOverlayText}>Drop a Python file here</span>
            <span className={styles.dragOverlaySubtext}>Accepts .py source files to populate editor</span>
          </div>
        )}
        <div className={styles.lineNumbers} aria-hidden="true">
          {Array.from({ length: Math.max(lineCount, 1) }, (_, i) => (
            <span key={i + 1}>{i + 1}</span>
          ))}
        </div>
        <textarea
          ref={textareaRef}
          className={styles.textarea}
          value={value}
          onChange={e => {
            onChange(e.target.value);
            if (fileMessage) setFileMessage(null);
          }}
          onKeyDown={handleKeyDown}
          spellCheck={false}
          autoCorrect="off"
          autoCapitalize="off"
          placeholder={"def greet(name: str) -> str:\n    ...\n\n# Press Ctrl+Enter to generate docs or drag & drop a .py file here"}
          aria-label="Python source code input"
        />
      </div>
      {fileMessage && (
        <div
          className={`${styles.statusToast} ${
            fileMessage.type === 'error' ? styles.statusToastError : styles.statusToastSuccess
          }`}
          role="status"
        >
          <span>{fileMessage.text}</span>
          <button
            type="button"
            className={styles.statusToastDismiss}
            onClick={() => setFileMessage(null)}
            aria-label="Dismiss message"
          >
            ✕
          </button>
        </div>
      )}
      <div className={styles.footer}>
        <span className={styles.hint}>Ctrl+Enter to generate • Drag & drop .py file</span>
        <span className={styles.charCount}>{value.length.toLocaleString()} chars</span>
      </div>
    </Card>
  );
}
