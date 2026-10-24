import { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';
import { Card } from '../common/Card';
import { PillButton } from '../common/PillButton';
import styles from './MermaidCard.module.css';

// Initialize mermaid once
mermaid.initialize({
  startOnLoad: false,
  theme: 'base',
  themeVariables: {
    primaryColor: '#F1D5D1',
    primaryBorderColor: '#121210',
    primaryTextColor: '#121210',
    lineColor: '#121210',
    secondaryColor: '#FFFEFC',
    tertiaryColor: '#EF9793',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Inter', system-ui, sans-serif",
    fontSize: '14px',
  },
  flowchart: { curve: 'basis', padding: 20 },
});

let mermaidCounter = 0;

interface MermaidCardProps {
  code: string | null;
  loading: boolean;
}

export function MermaidCard({ code, loading }: MermaidCardProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [showRaw, setShowRaw] = useState(false);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [svgContent, setSvgContent] = useState<string | null>(null);

  useEffect(() => {
    if (!code) { setSvgContent(null); setRenderError(null); return; }

    const id = `mermaid-${++mermaidCounter}`;
    setRenderError(null);

    mermaid.render(id, code)
      .then(({ svg }) => {
        setSvgContent(svg);
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Mermaid render failed';
        setRenderError(msg);
        setSvgContent(null);
      });
  }, [code]);

  const headerActions = code ? (
    <PillButton size="sm" variant="ghost" onClick={() => setShowRaw(r => !r)}>
      {showRaw ? 'View Diagram' : 'View Source'}
    </PillButton>
  ) : undefined;

  if (loading) return null; // Don't show placeholder while generating

  if (!code) return null; // Only show when there is Mermaid output

  return (
    <Card title="Control Flow Diagram" headerActions={headerActions}>
      {showRaw ? (
        <pre className={styles.rawCode}>{code}</pre>
      ) : renderError ? (
        <div className={styles.error}>
          <span>⚠ Diagram render error</span>
          <span className={styles.errorDetail}>{renderError}</span>
        </div>
      ) : svgContent ? (
        <div
          ref={containerRef}
          className={styles.svgWrap}
          dangerouslySetInnerHTML={{ __html: svgContent }}
        />
      ) : (
        <div className={styles.placeholder}>Rendering diagram…</div>
      )}
    </Card>
  );
}
