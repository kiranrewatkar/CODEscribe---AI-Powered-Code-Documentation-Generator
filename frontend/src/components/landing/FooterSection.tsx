import styles from './FooterSection.module.css';

interface FooterSectionProps {
  onOpenWorkspace: () => void;
  onReplayIntro?: () => void;
}

export function FooterSection({ onOpenWorkspace, onReplayIntro }: FooterSectionProps) {
  return (
    <footer id="about" className={styles.footer} role="contentinfo">
      <div className={styles.container}>
        <div className={styles.topRow}>
          {/* Brand Info */}
          <div className={styles.brandCol}>
            <div className={styles.brandLogo}>
              <span className={styles.logoBadge}>CS</span>
              <span className={styles.logoTitle}>CODEscribe</span>
            </div>
            <p className={styles.tagline}>
              Local, private Python API documentation and deterministic AST Mermaid
              flowcharts powered by fine-tuned CodeT5.
            </p>
            <div className={styles.academicBadge}>
              MCA Academic Project · Local Inference Only
            </div>
          </div>

          {/* Navigation Links Column */}
          <div className={styles.linkCol}>
            <h4 className={styles.colTitle}>Navigation</h4>
            <ul className={styles.linkList}>
              <li><a href="#overview">Overview</a></li>
              <li><a href="#how-it-works">How it works</a></li>
              <li><a href="#process">Pipeline</a></li>
              <li><a href="#stats">Benchmarks</a></li>
              <li><a href="#workspace" onClick={onOpenWorkspace}>Workspace</a></li>
              <li><a href="#faq">FAQs</a></li>
            </ul>
          </div>

          {/* Project Architecture Column */}
          <div className={styles.linkCol}>
            <h4 className={styles.colTitle}>Architecture</h4>
            <ul className={styles.linkList}>
              <li><span>Model: Salesforce/codet5-small</span></li>
              <li><span>Parameters: 60,492,288</span></li>
              <li><span>Backend: FastAPI + Uvicorn</span></li>
              <li><span>Frontend: React 19 + TypeScript</span></li>
              <li><span>Diagrams: AST + Mermaid.js</span></li>
            </ul>
          </div>

          {/* Actions & Links Column */}
          <div className={styles.linkCol}>
            <h4 className={styles.colTitle}>Quick Actions</h4>
            <ul className={styles.linkList}>
              <li>
                <button type="button" className={styles.linkButton} onClick={onOpenWorkspace}>
                  Launch Workspace →
                </button>
              </li>
              {onReplayIntro && (
                <li>
                  <button type="button" className={styles.linkButton} onClick={onReplayIntro}>
                    Replay Cinematic Intro ▶
                  </button>
                </li>
              )}
              <li>
                <span className={styles.endpointLabel}>Local API: http://127.0.0.1:8000</span>
              </li>
            </ul>
          </div>
        </div>

        <div className={styles.bottomRow}>
          <div className={styles.copy}>
            © {new Date().getFullYear()} CODEscribe. Developed for Master of Computer Applications (MCA).
          </div>
          <div className={styles.privacyNote}>
            Private by design: Python code never transmits beyond your local host.
          </div>
        </div>
      </div>
    </footer>
  );
}
