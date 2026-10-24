import { useScrollReveal } from '../../hooks/useScrollReveal';
import styles from './ExplanationSection.module.css';

const STEPS = [
  { num: '1', title: 'Source Input', text: 'Paste any valid Python function or load from curated dataset examples.' },
  { num: '2', title: 'AST Validation', text: 'Abstract Syntax Tree parser extracts symbols, arguments, and verifies syntax.' },
  { num: '3', title: 'Model Inference', text: 'Fine-tuned CodeT5 predicts docstring semantics and Markdown formatting.' },
  { num: '4', title: 'Flowchart Generation', text: 'Control structures deterministically map to Mermaid flowchart nodes.' },
];

export function ExplanationSection() {
  const revealRef = useScrollReveal();

  return (
    <section id="how-it-works" className={styles.explanationSection}>
      <div ref={revealRef} className={`${styles.container} reveal-fade-up`}>
        <div className={styles.tag}>Architecture Overview</div>
        <h2 className={styles.heading}>How CODEscribe works</h2>
        <p className={styles.lead}>
          A seamless bridge connecting deep neural code summarization with deterministic
          structural compilation. CODEscribe automates documentation without black-box opacity.
        </p>

        <div className={styles.stepsGrid}>
          {STEPS.map((step) => (
            <div key={step.num} className={styles.stepCard}>
              <span className={styles.stepNum}>{step.num}</span>
              <h3 className={styles.stepTitle}>{step.title}</h3>
              <p className={styles.stepText}>{step.text}</p>
            </div>
          ))}
        </div>

        {/* Code-oriented visual strip */}
        <div className={styles.codeStrip} aria-hidden="true">
          <div className={styles.stripTrack}>
            <span className={styles.stripToken}>ast.parse(source)</span>
            <span className={styles.stripArrow}>→</span>
            <span className={styles.stripToken}>tokenizer(code, prefix='summarize: ')</span>
            <span className={styles.stripArrow}>→</span>
            <span className={styles.stripToken}>model.generate(num_beams=1)</span>
            <span className={styles.stripArrow}>→</span>
            <span className={styles.stripToken}>mermaid.render(ast_flowchart)</span>
          </div>
        </div>
      </div>
    </section>
  );
}
