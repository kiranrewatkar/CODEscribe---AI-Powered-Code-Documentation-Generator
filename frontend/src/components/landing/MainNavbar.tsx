import { useState, useEffect, useCallback } from 'react';
import { PillButton } from '../common/PillButton';
import styles from './MainNavbar.module.css';

interface MainNavbarProps {
  onOpenWorkspace: () => void;
  onReplayIntro?: () => void;
}

const NAV_LINKS = [
  { href: '#overview', label: 'Overview' },
  { href: '#how-it-works', label: 'How it works' },
  { href: '#process', label: 'Process' },
  { href: '#stats', label: 'Numbers' },
  { href: '#workspace', label: 'Workspace' },
  { href: '#faq', label: 'FAQs' },
];

export function MainNavbar({ onOpenWorkspace, onReplayIntro }: MainNavbarProps) {
  const [isSticky, setIsSticky] = useState(false);
  const [activeSection, setActiveSection] = useState('overview');

  useEffect(() => {
    let ticking = false;

    const handleScroll = () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          const scrollY = window.scrollY;
          setIsSticky(scrollY > 120);

          // Update active link based on scroll position
          const sections = NAV_LINKS.map((link) => link.href.substring(1));
          for (let i = sections.length - 1; i >= 0; i--) {
            const el = document.getElementById(sections[i]);
            if (el) {
              const rect = el.getBoundingClientRect();
              if (rect.top <= 200) {
                setActiveSection(sections[i]);
                break;
              }
            }
          }
          ticking = false;
        });
        ticking = true;
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();

    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleNavClick = useCallback(
    (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
      e.preventDefault();
      const targetId = href.substring(1);
      const targetEl = document.getElementById(targetId);
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: 'smooth' });
      }
    },
    []
  );

  return (
    <header
      className={`${styles.header} ${isSticky ? styles.isSticky : ''}`}
      role="banner"
    >
      <div className={styles.navbarInner}>
        {/* Brand */}
        <a href="#overview" className={styles.brand} onClick={(e) => handleNavClick(e, '#overview')}>
          <span className={styles.brandMark}>CS</span>
          <span className={styles.brandName}>CODEscribe</span>
        </a>

        {/* Links */}
        <nav className={styles.navLinks} aria-label="Main navigation">
          {NAV_LINKS.map((link) => {
            const id = link.href.substring(1);
            const isActive = activeSection === id;
            return (
              <a
                key={link.href}
                href={link.href}
                onClick={(e) => handleNavClick(e, link.href)}
                className={`${styles.navLink} ${isActive ? styles.navLinkActive : ''}`}
              >
                {link.label}
              </a>
            );
          })}
        </nav>

        {/* Actions */}
        <div className={styles.actions}>
          {onReplayIntro && (
            <PillButton
              variant="secondary"
              size="sm"
              onClick={onReplayIntro}
              title="Watch intro animation"
              aria-label="Replay intro animation"
            >
              Intro
            </PillButton>
          )}
          <PillButton
            variant="primary"
            size="sm"
            onClick={onOpenWorkspace}
          >
            Open workspace
          </PillButton>
        </div>
      </div>
    </header>
  );
}
