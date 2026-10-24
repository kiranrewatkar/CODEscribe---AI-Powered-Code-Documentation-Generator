import { useState, useCallback } from 'react';
import { PillButton } from '../common/PillButton';
import { useScrollReveal } from '../../hooks/useScrollReveal';
import styles from './FaqSection.module.css';

interface FaqItem {
  id: string;
  question: string;
  answer: string;
}

const FAQS: FaqItem[] = [
  {
    id: 'how-it-works',
    question: 'How does CODEscribe generate documentation?',
    answer:
      'CODEscribe takes raw Python source code, removes any existing docstrings, parses the AST to inspect control flow and parameter signatures, passes the normalized code to a fine-tuned CodeT5 model, and assembles the result into GitHub-flavored Markdown alongside an AST-generated Mermaid diagram.',
  },
  {
    id: 'languages-supported',
    question: 'Which programming language does the current version support?',
    answer:
      'The current model and AST pipeline are specialized specifically for Python (tested on Python 3.12). Support for additional languages like JavaScript or Rust may be explored in future milestones.',
  },
  {
    id: 'codet5-role',
    question: 'What role does CodeT5-small play?',
    answer:
      'CodeT5-small acts as the neural sequence-to-sequence summarizer. It has 60,492,288 parameters and was fine-tuned on CodeSearchNet Python function-docstring pairs using an encoder-decoder architecture with relative position encodings.',
  },
  {
    id: 'flowcharts-ast',
    question: 'How are flowcharts generated?',
    answer:
      'Flowcharts are created deterministically through an AST visitor written in pure Python (app/ast_mermaid.py). It walks statements, branch conditions (if/elif/else), loops (for, while), and exit returns to output valid Mermaid.js flowchart syntax without hallucination.',
  },
  {
    id: 'evaluation-metrics',
    question: 'How are the generated results evaluated?',
    answer:
      'The fully trained checkpoint was evaluated against a held-out test split of 993 eligible Python functions, achieving ROUGE-1 of 26.69%, ROUGE-2 of 8.70%, ROUGE-L of 24.36%, and BLEU of 4.85% under deterministic beam search.',
  },
  {
    id: 'review-required',
    question: 'Does the documentation need to be reviewed?',
    answer:
      'Yes. As with any LLM-assisted generation tool, generated docstrings should be reviewed by an engineer before committing them to production codebases, particularly for safety-critical systems.',
  },
];

interface FaqSectionProps {
  onOpenWorkspace: () => void;
}

export function FaqSection({ onOpenWorkspace }: FaqSectionProps) {
  const [openId, setOpenId] = useState<string | null>('how-it-works');
  const revealRef = useScrollReveal();

  const handleToggle = useCallback((id: string) => {
    setOpenId((prev) => (prev === id ? null : id));
  }, []);

  return (
    <section id="faq" className={styles.faqSection}>
      <div ref={revealRef} className={`${styles.container} reveal-fade-up`}>
        {/* Left Column */}
        <div className={styles.leftCol}>
          <span className={styles.tag}>Common Questions</span>
          <h2 className={styles.heading}>Frequently Asked Questions</h2>
          <p className={styles.lead}>
            Everything you need to know about CODEscribe’s local model, architecture,
            and documentation guarantees.
          </p>
          <PillButton
            variant="primary"
            onClick={onOpenWorkspace}
            className={styles.actionBtn}
          >
            Open Workspace →
          </PillButton>
        </div>

        {/* Right Column Accordion */}
        <div className={styles.accordionList} role="region" aria-label="Frequently Asked Questions Accordion">
          {FAQS.map((faq) => {
            const isOpen = openId === faq.id;
            return (
              <div
                key={faq.id}
                className={`${styles.faqRow} ${isOpen ? styles.isOpen : ''}`}
              >
                <button
                  type="button"
                  className={styles.questionBtn}
                  onClick={() => handleToggle(faq.id)}
                  aria-expanded={isOpen}
                  aria-controls={`faq-answer-${faq.id}`}
                >
                  <span className={styles.questionText}>{faq.question}</span>
                  <span className={styles.iconWrapper} aria-hidden="true">
                    <svg
                      className={styles.chevronIcon}
                      width="18"
                      height="18"
                      viewBox="0 0 24 24"
                      fill="none"
                    >
                      <path
                        d="M6 9L12 15L18 9"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </span>
                </button>

                <div
                  id={`faq-answer-${faq.id}`}
                  className={styles.answerRegion}
                  role="region"
                >
                  <div className={styles.answerInner}>
                    <p className={styles.answerText}>{faq.answer}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
