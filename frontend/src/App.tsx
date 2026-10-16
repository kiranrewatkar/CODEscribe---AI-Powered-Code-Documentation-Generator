import { useState, useCallback } from 'react';
import { HeroScreen } from './components/hero/HeroScreen';
import { WorkspaceView } from './components/workspace/WorkspaceView';
import { CinematicIntro } from './components/intro/CinematicIntro';
import './styles/global.css';

type AppView = 'intro' | 'hero' | 'workspace';
type WorkspaceTab = 'editor' | 'examples' | 'about';

const INTRO_STORAGE_KEY = 'codescribe_intro_seen';

function hasSeenIntro(): boolean {
  try {
    return localStorage.getItem(INTRO_STORAGE_KEY) === 'true';
  } catch {
    return false;
  }
}

function markIntroSeen(): void {
  try {
    localStorage.setItem(INTRO_STORAGE_KEY, 'true');
  } catch {
    // In-memory fallback if localStorage is restricted or disabled
  }
}

function App() {
  // Only play intro on first visit; otherwise start directly on hero screen
  const [view, setView] = useState<AppView>(() => {
    return hasSeenIntro() ? 'hero' : 'intro';
  });
  const [tab, setTab] = useState<WorkspaceTab>('editor');

  const handleEnterWorkspace = useCallback((initialTab: WorkspaceTab = 'editor') => {
    setTab(initialTab);
    setView('workspace');
  }, []);

  const handleIntroComplete = useCallback(() => {
    markIntroSeen();
    setView('hero');
  }, []);

  const handleReplayIntro = useCallback(() => {
    setView('intro');
  }, []);

  const handleBackToHero = useCallback(() => {
    setView('hero');
  }, []);

  return (
    <>
      {view === 'intro' && (
        <CinematicIntro onComplete={handleIntroComplete} />
      )}
      {view === 'hero' && (
        <HeroScreen
          onEnter={handleEnterWorkspace}
          onReplayIntro={handleReplayIntro}
        />
      )}
      {view === 'workspace' && (
        <WorkspaceView
          initialView={tab}
          onBack={handleBackToHero}
        />
      )}
    </>
  );
}

export default App;
