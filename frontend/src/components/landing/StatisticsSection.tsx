import { useScrollReveal } from '../../hooks/useScrollReveal';
import styles from './StatisticsSection.module.css';

const STATS = [
  {
    value: '60.49M',
    label: 'Model Parameters',
    sub: 'Salesforce/codet5-small fine-tuned for 3 epochs with AdamW on CodeSearchNet Python.',
  },
  {
    value: '993',
    label: 'Test Evaluation Samples',
    sub: 'Held-out test split evaluated after complete training run, rejecting invalid syntax and empty pairs.',
  },
  {
    value: '4',
    label: 'Evaluated Benchmark Metrics',
    sub: 'ROUGE-1: 26.69% · ROUGE-2: 8.70% · ROUGE-L: 24.36% · BLEU: 4.85% under deterministic beam search.',
  },
];

export function StatisticsSection() {
  const revealRef = useScrollReveal();

  return (
    <section id="stats" className={styles.statsSection}>
      {/* Stepped-Square Mosaic Transition Boundary at Top */}
      <div className={styles.mosaicBoundary} aria-hidden="true">
        <div className={styles.boundaryRow}>
          <span className={styles.bSq} style={{ opacity: 0.2 }} />
          <span className={styles.bSq} style={{ opacity: 0.5 }} />
          <span className={styles.bSq} style={{ opacity: 0.8 }} />
          <span className={styles.bSq} style={{ opacity: 0.3 }} />
          <span className={styles.bSq} style={{ opacity: 0.6 }} />
          <span className={styles.bSq} style={{ opacity: 0.9 }} />
          <span className={styles.bSq} style={{ opacity: 0.4 }} />
          <span className={styles.bSq} style={{ opacity: 0.7 }} />
          <span className={styles.bSq} style={{ opacity: 0.5 }} />
          <span className={styles.bSq} style={{ opacity: 0.8 }} />
          <span className={styles.bSq} style={{ opacity: 0.3 }} />
          <span className={styles.bSq} style={{ opacity: 0.6 }} />
        </div>
      </div>

      <div ref={revealRef} className={`${styles.container} reveal-fade-up`}>
        <div className={styles.headerRow}>
          <div className={styles.headingCol}>
            <span className={styles.tag}>Empirical Benchmark</span>
            <h2 className={styles.heading}>CODEscribe in numbers</h2>
          </div>
          <p className={styles.leadParagraph}>
            All metrics are derived directly from the held-out evaluation phase on local hardware.
            No simulated benchmarks or synthetic vanity metrics.
          </p>
        </div>

        <div className={styles.statsGrid}>
          {STATS.map((stat, idx) => (
            <div key={idx} className={`${styles.statCard} reveal-child`}>
              <div className={styles.statValue}>{stat.value}</div>
              <h3 className={styles.statLabel}>{stat.label}</h3>
              <p className={styles.statSub}>{stat.sub}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
