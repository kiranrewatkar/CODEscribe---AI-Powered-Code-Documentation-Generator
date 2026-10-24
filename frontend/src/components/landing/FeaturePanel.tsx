import { useScrollReveal } from '../../hooks/useScrollReveal';
import styles from './FeaturePanel.module.css';

export function FeaturePanel() {
  const revealRef = useScrollReveal();

  return (
    <section className={styles.sectionWrapper}>
      <div ref={revealRef} className={`${styles.charcoalPanel} reveal-fade-up`}>
        <div className={styles.textColumn}>
          <div className={styles.badge}>
            <span className={styles.badgeSquare} />
            <span>Dual Synthesis Pipeline</span>
          </div>
          <h2 className={styles.heading}>
            From raw code to a clearer understanding.
          </h2>
          <p className={styles.leadText}>
            Traditional docstring generators only produce text. CODEscribe pairs neural documentation
            synthesis with deterministic AST analysis to yield both human-readable Markdown specifications
            and renderable Mermaid flowcharts.
          </p>
          <ul className={styles.featureList}>
            <li>
              <strong>CodeT5-small Neural Synthesis:</strong> Trained specifically on Python functions to infer intent, parameters, and return types.
            </li>
            <li>
              <strong>Deterministic Control Flow:</strong> Exact visitor traverses `if/elif/else`, `for`, `while`, and return trees with stable IDs.
            </li>
            <li>
              <strong>Private & Local:</strong> Zero network hops. Your proprietary source code never leaves your local hardware.
            </li>
          </ul>
        </div>

        <div className={styles.visualColumn} aria-hidden="true">
          <div className={styles.diagramContainer}>
            {/* Input Node */}
            <div className={styles.nodeCard}>
              <div className={styles.nodeHeader}>
                <span className={styles.dotRed} />
                <span className={styles.dotYellow} />
                <span className={styles.dotGreen} />
                <span className={styles.nodeTitle}>source.py</span>
              </div>
              <pre className={styles.nodeCode}>
                <code>{`def parse_query(q):\n  return [w for w in q]`}</code>
              </pre>
            </div>

            <div className={styles.flowConnector}>
              <div className={styles.lineVertical} />
              <div className={styles.splitBranch}>
                <span className={styles.branchPill}>Neural</span>
                <span className={styles.branchPill}>AST Visitor</span>
              </div>
            </div>

            {/* Output Dual Nodes */}
            <div className={styles.outputGrid}>
              <div className={styles.outCard}>
                <div className={styles.outIcon}>📄</div>
                <div className={styles.outTitle}>Markdown Spec</div>
                <div className={styles.outSub}>Docstring + Types</div>
              </div>
              <div className={styles.outCard}>
                <div className={styles.outIcon}>🔀</div>
                <div className={styles.outTitle}>Mermaid Chart</div>
                <div className={styles.outSub}>Graph TD / Loops</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
