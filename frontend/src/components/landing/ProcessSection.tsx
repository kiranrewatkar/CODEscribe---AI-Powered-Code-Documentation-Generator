import { useScrollReveal } from '../../hooks/useScrollReveal';
import { usePointerParallax } from '../../hooks/usePointerParallax';
import styles from './ProcessSection.module.css';

const PROCESS_CARDS = [
  {
    num: '01',
    title: 'Input & AST Sanitization',
    desc: 'Python code is parsed into an abstract syntax tree. Docstrings are removed for unbiased evaluation and control flow statements are cataloged.',
  },
  {
    num: '02',
    title: 'Neural Docstring Synthesis',
    desc: 'The fine-tuned CodeT5 encoder-decoder transformer (60.5M parameters) consumes the tokenized body and synthesizes an accurate natural language summary.',
  },
  {
    num: '03',
    title: 'Markdown Specification Generator',
    desc: 'The synthesized summary is structured with parameter types, return values, exceptions, and usage examples formatted in GitHub-compatible Markdown.',
  },
  {
    num: '04',
    title: 'Deterministic Mermaid Flowchart',
    desc: 'An AST visitor walks sequential statements, conditionals, and loops to generate clean, reproducible graph syntax rendered directly into SVG.',
  },
];

export function ProcessSection() {
  const revealRef = useScrollReveal();
  const parallaxRef = usePointerParallax<HTMLElement>();

  return (
    <section id="process" ref={parallaxRef} className={styles.processSection}>
      <div ref={revealRef} className={`${styles.container} reveal-fade-up`}>
        <div className={styles.sectionHeader}>
          <span className={styles.tag}>Visual Pipeline</span>
          <h2 className={styles.heading}>How code transforms into clarity</h2>
        </div>

        <div className={styles.asymmetricGrid}>
          {/* Left Column: Large Visual Process Illustration */}
          <div className={styles.illustrationWrapper}>
            <div className={styles.illustrationCard}>
              <div className={styles.cardGlow} aria-hidden="true" />

              {/* Animated visual diagram */}
              <div className={styles.flowStages}>
                <div className={styles.stageBlock}>
                  <div className={styles.stageIcon}>💻</div>
                  <div className={styles.stageLabel}>Python Source</div>
                  <div className={styles.stageSub}>def func(x, y)...</div>
                </div>

                <div className={styles.stageArrow}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <path d="M12 5V19M12 19L19 12M12 19L5 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </div>

                <div className={styles.stageBlockActive}>
                  <div className={styles.stageIcon}>🧠</div>
                  <div className={styles.stageLabel}>CodeT5 + AST Visitor</div>
                  <div className={styles.stageSub}>Dual Engine Processing</div>
                </div>

                <div className={styles.stageArrow}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <path d="M12 5V19M12 19L19 12M12 19L5 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </div>

                <div className={styles.stageOutputs}>
                  <div className={styles.outMini}>
                    <span>📝</span> Markdown Spec
                  </div>
                  <div className={styles.outMini}>
                    <span>📊</span> Mermaid Diagram
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Explanatory Cards */}
          <div className={styles.cardsColumn}>
            {PROCESS_CARDS.map((card) => (
              <div key={card.num} className={`${styles.processCard} reveal-child`}>
                <div className={styles.cardHeader}>
                  <span className={styles.cardNum}>{card.num}</span>
                  <h3 className={styles.cardTitle}>{card.title}</h3>
                </div>
                <p className={styles.cardDesc}>{card.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
