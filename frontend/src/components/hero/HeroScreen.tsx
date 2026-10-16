import { PillNavbar } from '../navigation/PillNavbar';
import styles from './HeroScreen.module.css';

interface HeroScreenProps {
  onEnter: (view?: 'editor' | 'examples' | 'about') => void;
  onReplayIntro?: () => void;
}

export function HeroScreen({ onEnter, onReplayIntro }: HeroScreenProps) {
  return (
    <div className={styles.hero}>
      {/* Background gradient layers */}
      <div className={styles.gradientLayer} aria-hidden="true" />
      <div className={styles.gridOverlay} aria-hidden="true" />

      {/* Top navigation with Reference Image 2 Pill Navbar */}
      <header className={styles.header}>
        <div className={styles.logo}>
          <span className={styles.logoMark}>CS</span>
          <span className={styles.logoText}>CODEscribe</span>
        </div>
        <PillNavbar
          items={[
            { id: 'home', label: 'Home' },
            { id: 'editor', label: 'Workspace' },
            { id: 'examples', label: 'Examples' },
            { id: 'about', label: 'About' },
          ]}
          activeId="home"
          onSelect={(id) => {
            if (id === 'home') return;
            onEnter(id as 'editor' | 'examples' | 'about');
          }}
          actionButton={{
            label: 'Launch App',
            onClick: () => onEnter('editor'),
          }}
        />
        <div className={styles.headerBadge}>
          <span className={styles.headerBadgeDot} />
          <span>Local GPU</span>
        </div>
      </header>

      <div className={styles.content}>
        <div className={styles.badge}>
          <span className={styles.badgeDot} />
          AI-Powered Documentation
        </div>

        <h1 className={styles.title}>
          <span className={styles.titleMain}>CODEscribe</span>
          <span className={styles.titleSub}>
            Automated API Documentation
            <br />
            from Python Source Code
          </span>
        </h1>

        <p className={styles.description}>
          Fine-tuned CodeT5 model running locally on your GPU.
          Paste your Python functions and receive structured Markdown
          documentation and AST-based Mermaid flowcharts — instantly.
        </p>

        <div className={styles.actions}>
          <div className={styles.ctaRow}>
            <button className={styles.ctaButton} onClick={() => onEnter('editor')}>
              Open Workspace
              <span className={styles.ctaArrow} aria-hidden="true">→</span>
            </button>
            {onReplayIntro && (
              <button
                className={styles.replayButton}
                onClick={onReplayIntro}
                aria-label="Replay intro animation"
                title="Watch the code-to-documentation intro animation again"
              >
                <span className={styles.replayIcon} aria-hidden="true">▶</span>
                <span>Replay Intro</span>
              </button>
            )}
          </div>
          <div className={styles.metaBadges}>
            <span className={styles.metaBadge}>CodeT5-small</span>
            <span className={styles.metaBadge}>ROUGE-1 26.69%</span>
            <span className={styles.metaBadge}>Local · Private</span>
          </div>
        </div>
      </div>

      {/* Decorative corner element */}
      <div className={styles.cornerDecor} aria-hidden="true">
        <svg width="320" height="320" viewBox="0 0 320 320" fill="none">
          <circle cx="160" cy="160" r="120" stroke="rgba(201,168,76,0.08)" strokeWidth="1" />
          <circle cx="160" cy="160" r="80" stroke="rgba(201,168,76,0.12)" strokeWidth="1" />
          <circle cx="160" cy="160" r="40" stroke="rgba(201,168,76,0.18)" strokeWidth="1" />
          <circle cx="160" cy="160" r="4" fill="rgba(201,168,76,0.4)" />
        </svg>
      </div>
    </div>
  );
}
