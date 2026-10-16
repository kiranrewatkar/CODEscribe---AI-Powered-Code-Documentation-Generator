import { useState, useEffect, useCallback, useRef } from 'react';
import styles from './CinematicIntro.module.css';

interface CinematicIntroProps {
  onComplete: () => void;
}

type Stage = 'code' | 'transform' | 'mermaid' | 'resolve';

const STAGE_DURATIONS = {
  code: 2800,      // Stage 1: Floating code cards in 3D space
  transform: 2800, // Stage 2: Code transforms to Markdown documentation
  mermaid: 2600,   // Stage 3: Mermaid AST flowchart motif emerges
  resolve: 800,    // Stage 4: Fade & converge into Hero
};

export function CinematicIntro({ onComplete }: CinematicIntroProps) {
  const [stage, setStage] = useState<Stage>('code');
  const [progress, setProgress] = useState(0);
  const startTimeRef = useRef<number>(Date.now());
  const timerRef = useRef<number | null>(null);

  // Check for prefers-reduced-motion
  useEffect(() => {
    try {
      const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
      if (mediaQuery.matches) {
        onComplete();
      }
    } catch {
      // Fallback if matchMedia is unavailable
    }
  }, [onComplete]);

  // Handle skip with Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onComplete();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onComplete]);

  // Timeline progression
  useEffect(() => {
    const totalDuration =
      STAGE_DURATIONS.code +
      STAGE_DURATIONS.transform +
      STAGE_DURATIONS.mermaid +
      STAGE_DURATIONS.resolve;

    startTimeRef.current = Date.now();

    const interval = window.setInterval(() => {
      const elapsed = Date.now() - startTimeRef.current;
      const pct = Math.min(100, Math.round((elapsed / totalDuration) * 100));
      setProgress(pct);

      if (elapsed < STAGE_DURATIONS.code) {
        setStage('code');
      } else if (elapsed < STAGE_DURATIONS.code + STAGE_DURATIONS.transform) {
        setStage('transform');
      } else if (
        elapsed <
        STAGE_DURATIONS.code + STAGE_DURATIONS.transform + STAGE_DURATIONS.mermaid
      ) {
        setStage('mermaid');
      } else if (elapsed < totalDuration) {
        setStage('resolve');
      } else {
        window.clearInterval(interval);
        onComplete();
      }
    }, 40);

    timerRef.current = interval;
    return () => window.clearInterval(interval);
  }, [onComplete]);

  const handleSkip = useCallback(() => {
    if (timerRef.current) window.clearInterval(timerRef.current);
    onComplete();
  }, [onComplete]);

  return (
    <div className={`${styles.container} ${stage === 'resolve' ? styles.resolving : ''}`} role="region" aria-label="Intro animation">
      {/* Background atmospheric layers */}
      <div className={styles.bgGlowTeal} aria-hidden="true" />
      <div className={styles.bgGlowGold} aria-hidden="true" />
      <div className={styles.starGrid} aria-hidden="true" />

      {/* Top Bar with Skip Intro Control */}
      <header className={styles.topbar}>
        <div className={styles.logoBadge}>
          <span className={styles.logoDot} />
          <span className={styles.logoText}>CODEscribe</span>
        </div>
        <button
          className={styles.skipButton}
          onClick={handleSkip}
          aria-label="Skip introductory animation"
          title="Press Esc to skip"
        >
          <span>Skip Intro</span>
          <span className={styles.skipKey} aria-hidden="true">ESC</span>
        </button>
      </header>

      {/* 3D Scene Viewport */}
      <div className={styles.viewport}>
        {/* Stage 1: Floating layered Python code panels */}
        <div className={`${styles.sceneStage} ${styles.stageCode} ${stage === 'code' ? styles.activeStage : stage === 'transform' ? styles.leavingStage : styles.hiddenStage}`}>
          <div className={styles.perspectiveStage}>
            {/* Background floating snippet */}
            <div className={`${styles.codeCard} ${styles.codeCardBack}`} aria-hidden="true">
              <div className={styles.cardHeader}>
                <span className={styles.cardDotRed} />
                <span className={styles.cardDotYellow} />
                <span className={styles.cardDotGreen} />
                <span className={styles.cardFilename}>binary_search.py</span>
              </div>
              <pre className={styles.codeText}>
                <code>
                  <span className={styles.kw}>def</span> <span className={styles.fn}>binary_search</span>(arr: <span className={styles.type}>list</span>, t: <span className={styles.type}>int</span>) -&gt; <span className={styles.type}>int</span>:
                  {'\n'}    low, high = <span className={styles.num}>0</span>, <span className={styles.builtin}>len</span>(arr) - <span className={styles.num}>1</span>
                  {'\n'}    <span className={styles.kw}>while</span> low &lt;= high:
                  {'\n'}        mid = (low + high) // <span className={styles.num}>2</span>
                </code>
              </pre>
            </div>

            {/* Foreground main code snippet */}
            <div className={`${styles.codeCard} ${styles.codeCardFront}`}>
              <div className={styles.cardHeader}>
                <span className={styles.cardDotRed} />
                <span className={styles.cardDotYellow} />
                <span className={styles.cardDotGreen} />
                <span className={styles.cardFilename}>fibonacci.py</span>
                <span className={styles.cardTag}>Python AST</span>
              </div>
              <pre className={styles.codeText}>
                <code>
                  <span className={styles.kw}>def</span> <span className={styles.fn}>fibonacci</span>(n: <span className={styles.type}>int</span>) -&gt; <span className={styles.type}>int</span>:
                  {'\n'}    <span className={styles.kw}>if</span> n &lt;= <span className={styles.num}>0</span>:
                  {'\n'}        <span className={styles.kw}>return</span> <span className={styles.num}>0</span>
                  {'\n'}    <span className={styles.kw}>elif</span> n == <span className={styles.num}>1</span>:
                  {'\n'}        <span className={styles.kw}>return</span> <span className={styles.num}>1</span>
                  {'\n'}    a, b = <span className={styles.num}>0</span>, <span className={styles.num}>1</span>
                  {'\n'}    <span className={styles.kw}>for</span> _ <span className={styles.kw}>in</span> <span className={styles.builtin}>range</span>(<span className={styles.num}>2</span>, n + <span className={styles.num}>1</span>):
                  {'\n'}        a, b = b, a + b
                  {'\n'}    <span className={styles.kw}>return</span> b
                </code>
              </pre>
              <div className={styles.scanLine} aria-hidden="true" />
            </div>

            {/* Depth label */}
            <div className={styles.stageLabel}>
              <span className={styles.stepNum}>01</span>
              <span>Raw Source Ingestion &amp; AST Parsing</span>
            </div>
          </div>
        </div>

        {/* Stage 2: Code Morphing into Markdown Documentation */}
        <div className={`${styles.sceneStage} ${styles.stageTransform} ${stage === 'transform' ? styles.activeStage : stage === 'mermaid' ? styles.leavingStage : styles.hiddenStage}`}>
          <div className={styles.perspectiveStage}>
            <div className={styles.morphCard}>
              <div className={styles.cardHeaderDoc}>
                <span className={styles.docBadge}>CodeT5 Synthesis</span>
                <span className={styles.cardFilename}>fibonacci.md</span>
                <span className={styles.confidenceTag}>100% Local</span>
              </div>
              <div className={styles.docBody}>
                <h2 className={styles.docTitle}># `fibonacci`</h2>
                <div className={styles.docBlockquote}>
                  Computes the nth Fibonacci number iteratively with boundary condition checks.
                </div>
                <div className={styles.docSection}>
                  <h3 className={styles.docSubtitle}>Parameters:</h3>
                  <div className={styles.docRow}>
                    <code className={styles.docParam}>n (int)</code>
                    <span className={styles.docParamDesc}>Non-negative index of sequence.</span>
                  </div>
                </div>
                <div className={styles.docSection}>
                  <h3 className={styles.docSubtitle}>Returns:</h3>
                  <div className={styles.docRow}>
                    <code className={styles.docParam}>int</code>
                    <span className={styles.docParamDesc}>The calculated Fibonacci value.</span>
                  </div>
                </div>
              </div>
              <div className={styles.goldGlowBorder} aria-hidden="true" />
            </div>

            <div className={styles.stageLabel}>
              <span className={styles.stepNum}>02</span>
              <span>CodeT5 Generative Markdown Synthesis</span>
            </div>
          </div>
        </div>

        {/* Stage 3: Deterministic AST-to-Mermaid Flowchart Generation */}
        <div className={`${styles.sceneStage} ${styles.stageMermaid} ${(stage === 'mermaid' || stage === 'resolve') ? styles.activeStage : styles.hiddenStage}`}>
          <div className={styles.perspectiveStage}>
            <div className={styles.mermaidCard}>
              <div className={styles.cardHeaderMermaid}>
                <span className={styles.mermaidBadge}>Control Flow Synthesis</span>
                <span className={styles.cardFilename}>AST Flowchart</span>
              </div>
              <div className={styles.mermaidDiagramMock}>
                {/* Visual flowchart nodes motif */}
                <div className={styles.mermaidNodeStart}>Start: fibonacci(n)</div>
                <div className={styles.mermaidLine} />
                <div className={styles.mermaidNodeCond}>n &lt;= 0 ?</div>
                <div className={styles.mermaidBranches}>
                  <div className={styles.mermaidBranch}>
                    <span className={styles.branchLabel}>True</span>
                    <div className={styles.mermaidLineShort} />
                    <div className={styles.mermaidNodeAction}>return 0</div>
                  </div>
                  <div className={styles.mermaidBranch}>
                    <span className={styles.branchLabel}>False</span>
                    <div className={styles.mermaidLineShort} />
                    <div className={styles.mermaidNodeLoop}>for _ in range(2, n+1)</div>
                    <div className={styles.mermaidLineShort} />
                    <div className={styles.mermaidNodeReturn}>return b</div>
                  </div>
                </div>
              </div>
              <div className={styles.cyanGlowBorder} aria-hidden="true" />
            </div>

            <div className={styles.stageLabel}>
              <span className={styles.stepNum}>03</span>
              <span>Deterministic AST Control Flow Diagram</span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Timeline Indicator */}
      <footer className={styles.bottomBar}>
        <div className={styles.stageIndicators}>
          <div className={`${styles.indicator} ${stage === 'code' ? styles.indicatorActive : styles.indicatorPassed}`}>
            <span className={styles.indicatorNum}>1</span>
            <span className={styles.indicatorText}>Code Parsing</span>
          </div>
          <div className={styles.indicatorDivider} />
          <div className={`${styles.indicator} ${stage === 'transform' ? styles.indicatorActive : (stage === 'mermaid' || stage === 'resolve') ? styles.indicatorPassed : ''}`}>
            <span className={styles.indicatorNum}>2</span>
            <span className={styles.indicatorText}>Doc Synthesis</span>
          </div>
          <div className={styles.indicatorDivider} />
          <div className={`${styles.indicator} ${(stage === 'mermaid' || stage === 'resolve') ? styles.indicatorActive : ''}`}>
            <span className={styles.indicatorNum}>3</span>
            <span className={styles.indicatorText}>Mermaid Flow</span>
          </div>
        </div>

        <div className={styles.progressBarTrack} role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
          <div className={styles.progressBarFill} style={{ width: `${progress}%` }} />
        </div>
      </footer>
    </div>
  );
}
