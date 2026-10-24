import { useScrollReveal } from '../../hooks/useScrollReveal';
import styles from './TechStrip.module.css';

const TECHNOLOGIES = [
  { name: 'Python 3.12', detail: 'AST & Parsing' },
  { name: 'CodeT5-small', detail: 'Local Neural Model' },
  { name: 'FastAPI', detail: 'Inference Server' },
  { name: 'React 19', detail: 'Component Architecture' },
  { name: 'TypeScript', detail: 'Strict Client Contracts' },
  { name: 'Mermaid.js', detail: 'SVG Flowcharts' },
];

export function TechStrip() {
  const revealRef = useScrollReveal();

  return (
    <section className={styles.techSection}>
      <div ref={revealRef} className={`${styles.container} reveal-fade-up`}>
        <span className={styles.leadLabel}>Engineered with verified local open-source foundations</span>
        <div className={styles.techRow}>
          {TECHNOLOGIES.map((tech) => (
            <div key={tech.name} className={styles.techItem}>
              <span className={styles.techDot} />
              <div className={styles.techText}>
                <strong className={styles.techName}>{tech.name}</strong>
                <span className={styles.techDetail}>{tech.detail}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
