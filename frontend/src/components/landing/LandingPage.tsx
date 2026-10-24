import { useCallback } from 'react';
import { MainNavbar } from './MainNavbar';
import { HeroSection } from './HeroSection';
import { TechStrip } from './TechStrip';
import { FeaturePanel } from './FeaturePanel';
import { ExplanationSection } from './ExplanationSection';
import { ProcessSection } from './ProcessSection';
import { StatisticsSection } from './StatisticsSection';
import { WorkspaceSection } from './WorkspaceSection';
import { FaqSection } from './FaqSection';
import { FooterSection } from './FooterSection';
import { useHealth } from '../../hooks/useHealth';
import styles from './LandingPage.module.css';

interface LandingPageProps {
  onReplayIntro?: () => void;
  onEnterStandalone?: (tab?: 'editor' | 'examples' | 'about') => void;
}

export function LandingPage({ onReplayIntro, onEnterStandalone }: LandingPageProps) {
  const { health } = useHealth();

  const handleOpenWorkspace = useCallback(() => {
    const el = document.getElementById('workspace');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  }, []);

  return (
    <div className={styles.pageContainer}>
      {/* Sticky / Morphing Floating Navigation */}
      <MainNavbar
        onOpenWorkspace={handleOpenWorkspace}
        onReplayIntro={onReplayIntro}
      />

      <main>
        {/* Section 1: Hero & Quick Start */}
        <HeroSection
          onOpenWorkspace={handleOpenWorkspace}
          gpuName={health?.gpu_name}
          modelStatus={health?.status}
        />

        {/* Section 2: Technology & Trust Strip */}
        <TechStrip />

        {/* Section 3: Dark Feature Story Panel */}
        <FeaturePanel />

        {/* Section 4: Centered Explanation Section */}
        <ExplanationSection />

        {/* Section 5: Process Illustration & Sequential Explanatory Cards */}
        <ProcessSection />

        {/* Section 6: Coral Statistics Section */}
        <StatisticsSection />

        {/* Section 7: The Real Documentation Workspace */}
        <WorkspaceSection />

        {/* Section 8: FAQ Accordion */}
        <FaqSection onOpenWorkspace={handleOpenWorkspace} />
      </main>

      {/* Section 9: Dark Footer */}
      <FooterSection
        onOpenWorkspace={onEnterStandalone ? () => onEnterStandalone('editor') : handleOpenWorkspace}
        onReplayIntro={onReplayIntro}
      />
    </div>
  );
}
