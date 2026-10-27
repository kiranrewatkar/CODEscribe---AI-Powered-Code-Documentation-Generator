import { useState, useEffect, useCallback } from 'react';
import { Sidebar } from './Sidebar';
import { CodeEditor } from './CodeEditor';
import { DocViewer } from './DocViewer';
import { MermaidCard } from './MermaidCard';
import { ExamplesPanel } from './ExamplesPanel';
import { OptimizationPanel } from './OptimizationPanel';
import { StatusBadge } from '../common/StatusBadge';
import { PillNavbar } from '../navigation/PillNavbar';
import { useDocumentAPI } from '../../hooks/useDocumentAPI';
import { useOptimizeAPI } from '../../hooks/useOptimizeAPI';
import { useHealth } from '../../hooks/useHealth';
import { fetchExamples } from '../../api/client';
import type { ExampleSnippet } from '../../api/types';
import styles from './WorkspaceView.module.css';

type SidebarView = 'editor' | 'examples' | 'about';

const SIDEBAR_ITEMS = [
  { id: 'editor',   label: 'Editor',    icon: '✏' },
  { id: 'examples', label: 'Examples',  icon: '⚡' },
  { id: 'about',    label: 'About',     icon: 'ℹ' },
];

interface WorkspaceViewProps {
  initialView?: SidebarView;
  onBack: () => void;
}

export function WorkspaceView({ initialView = 'editor', onBack }: WorkspaceViewProps) {
  const [activeView, setActiveView] = useState<SidebarView>(initialView);
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

  // Load examples on mount
  useEffect(() => {
    fetchExamples()
      .then(r => setExamples(r.examples))
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

  const handleLoadExample = useCallback((exCode: string) => {
    setCode(exCode);
    setActiveView('editor');
    reset();
    resetOptimize();
  }, [reset, resetOptimize]);

  const handleSettingsChange = useCallback((s: Partial<{
    numBeams: number; maxNewTokens: number; includeMermaid: boolean
  }>) => {
    if (s.numBeams !== undefined) setNumBeams(s.numBeams);
    if (s.maxNewTokens !== undefined) setMaxNewTokens(s.maxNewTokens);
    if (s.includeMermaid !== undefined) setIncludeMermaid(s.includeMermaid);
  }, []);

  // Compute health badge status
  const healthStatus: 'ok' | 'degraded' | 'loading' | 'error' =
    healthLoading ? 'loading'
    : healthError  ? 'error'
    : health?.status === 'ok' ? 'ok'
    : 'degraded';

  return (
    <div className={styles.workspace}>
      {/* Sidebar matching Reference Image 3 */}
      <Sidebar
        items={SIDEBAR_ITEMS}
        activeId={activeView}
        onSelect={id => setActiveView(id as SidebarView)}
      />

      {/* Main area */}
      <div className={styles.main}>
        {/* Topbar with PillNavbar matching Reference Image 2 */}
        <header className={styles.topbar}>
          <div className={styles.topbarLeft}>
            <button className={styles.backBtn} onClick={onBack} title="Back to hero">
              ← Home
            </button>
          </div>
          <div className={styles.topbarCenter}>
            <PillNavbar
              items={[
                { id: 'editor', label: 'Editor' },
                { id: 'examples', label: 'Examples' },
                { id: 'about', label: 'About' },
              ]}
              activeId={activeView}
              onSelect={id => setActiveView(id as SidebarView)}
            />
          </div>
          <div className={styles.topbarRight}>
            <StatusBadge status={healthStatus} />
            {health?.gpu_name && (
              <span className={styles.gpuBadge} title={`VRAM: ${health.vram_allocated_mb?.toFixed(0)} / ${health.vram_total_mb?.toFixed(0)} MB`}>
                🎮 {health.gpu_name.replace('NVIDIA GeForce ', '')}
              </span>
            )}
          </div>
        </header>

        {/* Content area */}
        <div className={styles.content}>
          {activeView === 'editor' && (
            <div className={styles.editorLayout}>
              {/* Left: code editor */}
              <div className={styles.editorCol}>
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

              {/* Right: output */}
              <div className={styles.outputCol}>
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

          {activeView === 'examples' && (
            <div className={styles.singleCol}>
              <ExamplesPanel
                examples={examples}
                loading={examplesLoading}
                onSelect={handleLoadExample}
              />
            </div>
          )}

          {activeView === 'about' && (
            <div className={styles.singleCol}>
              <div className={styles.aboutCard}>
                <h2>About CODEscribe</h2>
                <p>
                  CODEscribe is an LLM-based Python API documentation generator fine-tuned
                  from <strong>Salesforce/codet5-small</strong> on the CodeSearchNet Python dataset.
                </p>
                <ul>
                  <li>Model: <code>model/codescribe</code> (60.5 M parameters)</li>
                  <li>Training: 3 epochs, CodeSearchNet Python (7,200 train pairs)</li>
                  <li>Evaluation: ROUGE-1 26.69% · ROUGE-2 8.70% · BLEU 4.85%</li>
                  <li>Backend: FastAPI + Uvicorn, local inference only</li>
                  <li>Privacy: your code never leaves your machine</li>
                </ul>
                {health && (
                  <div className={styles.healthDetails}>
                    <p><strong>PyTorch:</strong> {health.torch_version}</p>
                    <p><strong>Transformers:</strong> {health.transformers_version}</p>
                    <p><strong>Device:</strong> {health.device}</p>
                    {health.gpu_name && <p><strong>GPU:</strong> {health.gpu_name}</p>}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
