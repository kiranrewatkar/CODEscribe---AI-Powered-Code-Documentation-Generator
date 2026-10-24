import { useScrollReveal } from '../../hooks/useScrollReveal';
import { usePointerParallax } from '../../hooks/usePointerParallax';
import { AmbientBackground } from './AmbientBackground';
import styles from './HeroSection.module.css';

interface HeroSectionProps {
  onOpenWorkspace: () => void;
  gpuName?: string | null;
  modelStatus?: string;
}

export function HeroSection({ onOpenWorkspace, gpuName, modelStatus = 'ok' }: HeroSectionProps) {
  const revealRef = useScrollReveal();
  const parallaxRef = usePointerParallax<HTMLElement>();

  return (
    <section id="overview" ref={parallaxRef} className={styles.heroSection}>
      {/* Living Ambient Background with Gradient Drift & Parallax */}
      <AmbientBackground variant="hero" />

      {/* Geometric Stepped-Square Mosaic - Left Edge */}
      <div className={`${styles.mosaic} ${styles.mosaicLeft}`} aria-hidden="true">
        <div className={styles.sqCol}>
          <span className={styles.sq} style={{ opacity: 0.35 }} />
          <span className={styles.sq} style={{ opacity: 0.5 }} />
          <span className={styles.sq} style={{ opacity: 0.25 }} />
          <span className={styles.sq} style={{ opacity: 0.6 }} />
        </div>
        <div className={styles.sqCol}>
          <span className={styles.sq} style={{ opacity: 0.4 }} />
          <span className={styles.sq} style={{ opacity: 0.7 }} />
          <span className={styles.sq} style={{ opacity: 0.45 }} />
          <span className={styles.sq} style={{ opacity: 0.3 }} />
          <span className={styles.sq} style={{ opacity: 0.55 }} />
        </div>
        <div className={styles.sqCol}>
          <span className={styles.sq} style={{ opacity: 0.5 }} />
          <span className={styles.sq} style={{ opacity: 0.25 }} />
          <span className={styles.sq} style={{ opacity: 0.65 }} />
        </div>
        <div className={styles.sqCol}>
          <span className={styles.sq} style={{ opacity: 0.3 }} />
          <span className={styles.sq} style={{ opacity: 0.45 }} />
        </div>
      </div>

      {/* Geometric Stepped-Square Mosaic - Right Edge */}
      <div className={`${styles.mosaic} ${styles.mosaicRight}`} aria-hidden="true">
        <div className={styles.sqCol}>
          <span className={styles.sq} style={{ opacity: 0.35 }} />
          <span className={styles.sq} style={{ opacity: 0.6 }} />
        </div>
        <div className={styles.sqCol}>
          <span className={styles.sq} style={{ opacity: 0.5 }} />
          <span className={styles.sq} style={{ opacity: 0.3 }} />
          <span className={styles.sq} style={{ opacity: 0.7 }} />
        </div>
        <div className={styles.sqCol}>
          <span className={styles.sq} style={{ opacity: 0.4 }} />
          <span className={styles.sq} style={{ opacity: 0.65 }} />
          <span className={styles.sq} style={{ opacity: 0.35 }} />
          <span className={styles.sq} style={{ opacity: 0.55 }} />
        </div>
        <div className={styles.sqCol}>
          <span className={styles.sq} style={{ opacity: 0.3 }} />
          <span className={styles.sq} style={{ opacity: 0.5 }} />
          <span className={styles.sq} style={{ opacity: 0.4 }} />
        </div>
      </div>

      <div ref={revealRef} className={`${styles.content} reveal-fade-up`}>
        {/* Top Eyebrow Badge */}
        <div className={styles.eyebrow}>
          <span className={styles.eyebrowDot} />
          <span>Local Python Documentation Generator</span>
        </div>

        {/* Display Headline */}
        <h1 className={styles.headline}>
          From source code<br />
          to documentation that makes sense.
        </h1>

        {/* Supporting description */}
        <p className={styles.description}>
          Fine-tuned CodeT5 running locally on your GPU. Paste your Python functions
          to produce structured Markdown documentation and AST-based Mermaid flowcharts —
          instantly and privately on your machine.
        </p>

        {/* Floating White Hero Panel */}
        <div className={styles.heroPanel}>
          <div className={styles.panelHeader}>
            <div className={styles.panelBadge}>
              <span className={styles.badgePulse} />
              <span>Model Ready · Local GPU</span>
            </div>
            <div className={styles.panelMeta}>
              {gpuName ? (
                <span className={styles.gpuTag}>⚡ {gpuName.replace('NVIDIA GeForce ', '')}</span>
              ) : (
                <span className={styles.gpuTag}>⚡ Local Device</span>
              )}
              <span className={styles.tag}>CodeT5-small (60.5M)</span>
            </div>
          </div>

          <div className={styles.composerPreview}>
            <div className={styles.codeColumn}>
              <div className={styles.columnLabel}>Python Source</div>
              <pre className={styles.sampleCode}>
                <code>{`def compute_discount(price: float, rate: float = 0.1) -> float:
    """Validate inputs and calculate final discounted price."""
    if price < 0 or not (0 <= rate <= 1):
        raise ValueError("Invalid price or discount rate")
    return round(price * (1.0 - rate), 2)`}</code>
              </pre>
            </div>

            <div className={styles.dividerArrow} aria-hidden="true">
              →
            </div>

            <div className={styles.docColumn}>
              <div className={styles.columnLabel}>Generated Markdown & Flowchart</div>
              <div className={styles.sampleDoc}>
                <div className={styles.docSnippet}>
                  <h4>compute_discount</h4>
                  <p>Calculate the discounted price given a base price and discount rate.</p>
                  <div className={styles.miniDiagram}>
                    <span>[Start]</span> → <span>{'{Valid?}'}</span> → <span>[Return Price]</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className={styles.panelFooter}>
            <div className={styles.footerInfo}>
              <span className={styles.statusDot} data-status={modelStatus} />
              <span>Full control flow AST parsing + Markdown API format</span>
            </div>
            <button
              type="button"
              className={styles.openWorkspaceBtn}
              onClick={onOpenWorkspace}
            >
              Open Workspace
              <span className={styles.arrowIcon}>→</span>
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
