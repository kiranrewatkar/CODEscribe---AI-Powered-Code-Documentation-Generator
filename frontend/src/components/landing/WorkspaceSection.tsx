import { useState, useEffect, useCallback } from 'react';
import { CodeEditor } from '../workspace/CodeEditor';
import { DocViewer } from '../workspace/DocViewer';
import { MermaidCard } from '../workspace/MermaidCard';
import { ExamplesPanel } from '../workspace/ExamplesPanel';
import { OptimizationPanel } from '../workspace/OptimizationPanel';
import { StatusBadge } from '../common/StatusBadge';
import { useDocumentAPI } from '../../hooks/useDocumentAPI';
import { useOptimizeAPI } from '../../hooks/useOptimizeAPI';
import { useHealth } from '../../hooks/useHealth';
import { fetchExamples } from '../../api/client';
import { useScrollReveal } from '../../hooks/useScrollReveal';
import type { ExampleSnippet } from '../../api/types';
import styles from './WorkspaceSection.module.css';

type WorkspaceMode = 'editor' | 'examples' | 'about';

export function WorkspaceSection() {
  const [mode, setMode] = useState<WorkspaceMode>('editor');
  const [code, setCode] = useState('');
  const [examples, setExamples] = useState<ExampleSnippet[]>([]);
  const [examplesLoading, setExamplesLoading] = useState(true);

  // Generation settings
  const [numBeams, setNumBeams] = useState(1);
  const [maxNewTokens, setMaxNewTokens] = useState(128);
  const [includeMermaid, setIncludeMermaid] = useState(true);

  const { health, loading: healthLoading, error: healthError } = useHealth();
  const { generate, data: result, loading: generating, error: genError, reset } = useDocumentAPI();
  const { analyze: runOptimize, data: optimizeResult, loading: optimizing, error: optimizeError, reset: resetOptimize } = useOptimizeAPI();
  const revealRef = useScrollReveal();

  // Load examples on mount
  useEffect(() => {
    fetchExamples()
      .then((r) => {
        setExamples(r.examples);
        // Pre-populate with first example if editor is empty
        if (r.examples.length > 0 && !code) {
          setCode(r.examples[0].code);
        }
      })
      .catch(() => setExamples([]))
      .finally(() => setExamplesLoading(false));
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!code.trim()) return;
    await generate({
      code,
      num_beams: numBeams,
      max_new_tokens: maxNewTokens,
      temperature: 1.0,
      do_sample: false,
      include_mermaid: includeMermaid,
    });
  }, [code, numBeams, maxNewTokens, includeMermaid, generate]);

  const handleOptimize = useCallback(async () => {
    if (!code.trim()) return;
    await runOptimize(code);
  }, [code, runOptimize]);

  const handleLoadExample = useCallback(
    (exCode: string) => {
      setCode(exCode);
      setMode('editor');
      reset();
      resetOptimize();
    },
    [reset, resetOptimize]
  );

  const handleSettingsChange = useCallback(
    (s: Partial<{ numBeams: number; maxNewTokens: number; includeMermaid: boolean }>) => {
      if (s.numBeams !== undefined) setNumBeams(s.numBeams);
      if (s.maxNewTokens !== undefined) setMaxNewTokens(s.maxNewTokens);
      if (s.includeMermaid !== undefined) setIncludeMermaid(s.includeMermaid);
    },
    []
  );

  const healthStatus: 'ok' | 'degraded' | 'loading' | 'error' = healthLoading
    ? 'loading'
    : healthError
    ? 'error'
    : health?.status === 'ok'
    ? 'ok'
    : 'degraded';

  return (
    <section id="workspace" className={styles.workspaceSection}>
      <div ref={revealRef} className={`${styles.container} reveal-fade-up`}>
        {/* Workspace Section Header */}
        <div className={styles.sectionHeader}>
          <div className={styles.titleGroup}>
            <span className={styles.tag}>Live Documentation Generator</span>
            <h2 className={styles.heading}>Interactive Workspace</h2>
            <p className={styles.lead}>
              Input or choose Python source code below. Run full inference against the local
              fine-tuned model to generate structured documentation and control-flow diagrams.
            </p>
          </div>

          <div className={styles.controlsRow}>
            {/* View Mode Pill Switcher */}
            <div className={styles.modeTabs} role="tablist" aria-label="Workspace view modes">
              <button
                type="button"
                role="tab"
                aria-selected={mode === 'editor'}
                className={`${styles.modeTab} ${mode === 'editor' ? styles.modeTabActive : ''}`}
                onClick={() => setMode('editor')}
              >
                Editor
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={mode === 'examples'}
                className={`${styles.modeTab} ${mode === 'examples' ? styles.modeTabActive : ''}`}
                onClick={() => setMode('examples')}
              >
                Examples ({examples.length})
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={mode === 'about'}
                className={`${styles.modeTab} ${mode === 'about' ? styles.modeTabActive : ''}`}
                onClick={() => setMode('about')}
              >
                Model Info
              </button>
            </div>

            {/* Health & GPU indicators */}
            <div className={styles.healthStatusGroup}>
              <StatusBadge status={healthStatus} />
              {health?.gpu_name && (
                <span
                  className={styles.gpuBadge}
                  title={`VRAM: ${health.vram_allocated_mb?.toFixed(0)} / ${health.vram_total_mb?.toFixed(0)} MB`}
                >
                  ⚡ {health.gpu_name.replace('NVIDIA GeForce ', '')}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Content Views */}
        {mode === 'editor' && (
          <div className={styles.editorGrid}>
            {/* Left Column: Code Editor */}
            <div className={styles.editorPane}>
              <CodeEditor
                value={code}
                onChange={setCode}
                onSubmit={handleGenerate}
                loading={generating}
                examples={examples}
                numBeams={numBeams}
                maxNewTokens={maxNewTokens}
                includeMermaid={includeMermaid}
                onSettingsChange={handleSettingsChange}
                onOptimize={handleOptimize}
                optimizing={optimizing}
              />
            </div>

            {/* Right Column: Generated Documentation & Mermaid Diagram */}
            <div className={styles.outputPane}>
              {(optimizeResult || optimizing || optimizeError) && (
                <div style={{ marginBottom: '1.25rem' }}>
                  <OptimizationPanel
                    result={optimizeResult}
                    loading={optimizing}
                    error={optimizeError}
                    onClose={resetOptimize}
                  />
                </div>
              )}

              <DocViewer
                result={result}
                loading={generating}
                error={genError}
                onClearError={reset}
              />

              {(result?.mermaid_code || generating) && (
                <div className={styles.mermaidWrap}>
                  <MermaidCard
                    code={result?.mermaid_code ?? null}
                    loading={generating}
                  />
                </div>
              )}
            </div>
          </div>
        )}

        {mode === 'examples' && (
          <div className={styles.singlePane}>
            <ExamplesPanel
              examples={examples}
              loading={examplesLoading}
              onSelect={handleLoadExample}
            />
          </div>
        )}

        {mode === 'about' && (
          <div className={styles.singlePane}>
            <div className={styles.aboutCard}>
              <h3 className={styles.aboutTitle}>Salesforce/codet5-small fine-tuned on CodeSearchNet</h3>
              <p className={styles.aboutText}>
                CODEscribe is an end-to-end Python source-code documentation system developed
                for Master of Computer Applications (MCA). It integrates a locally deployed
                CodeT5 sequence-to-sequence model with a custom AST visitor to provide verifiable,
                syntax-aware documentation and flowcharts without external API latency or leakage.
              </p>

              <div className={styles.specsGrid}>
                <div className={styles.specItem}>
                  <span className={styles.specLabel}>Model Path</span>
                  <span className={styles.specVal}>model/codescribe</span>
                </div>
                <div className={styles.specItem}>
                  <span className={styles.specLabel}>Total Parameters</span>
                  <span className={styles.specVal}>60,492,288</span>
                </div>
                <div className={styles.specItem}>
                  <span className={styles.specLabel}>Training Split</span>
                  <span className={styles.specVal}>7,200 samples (3 epochs)</span>
                </div>
                <div className={styles.specItem}>
                  <span className={styles.specLabel}>Test Benchmark</span>
                  <span className={styles.specVal}>993 eligible samples</span>
                </div>
                <div className={styles.specItem}>
                  <span className={styles.specLabel}>ROUGE-1 / ROUGE-2</span>
                  <span className={styles.specVal}>26.69% / 8.70%</span>
                </div>
                <div className={styles.specItem}>
                  <span className={styles.specLabel}>ROUGE-L / BLEU</span>
                  <span className={styles.specVal}>24.36% / 4.85%</span>
                </div>
              </div>

              {health && (
                <div className={styles.runtimeBox}>
                  <h4>Runtime Environment</h4>
                  <div className={styles.runtimeDetails}>
                    <span>PyTorch: {health.torch_version}</span>
                    <span>Transformers: {health.transformers_version}</span>
                    <span>Device: {health.device}</span>
                    {health.gpu_name && <span>GPU: {health.gpu_name}</span>}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
