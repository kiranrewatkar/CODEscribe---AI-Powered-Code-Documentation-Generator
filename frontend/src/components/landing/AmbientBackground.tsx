import { InteractiveParticles } from './InteractiveParticles';
import styles from './AmbientBackground.module.css';

interface AmbientBackgroundProps {
  variant?: 'hero' | 'section';
}

export function AmbientBackground({ variant = 'hero' }: AmbientBackgroundProps) {
  return (
    <div className={`${styles.ambientRoot} ${styles[variant]}`} aria-hidden="true">
      {/* Soft Blurred Radial Atmosphere Orbs */}
      <div className={`${styles.orb} ${styles.orb1}`} />
      <div className={`${styles.orb} ${styles.orb2}`} />
      <div className={`${styles.orb} ${styles.orb3}`} />

      {/* Continuously moving particles that interact with cursor */}
      <InteractiveParticles themeMode={variant === 'hero' ? 'coral' : 'cream'} />

      {/* Floating Geometric Square Motifs */}
      <div className={`${styles.driftCluster} ${styles.clusterLeft}`}>
        <span className={styles.driftSq} style={{ animationDelay: '0s', width: 22, height: 22, opacity: 0.45 }} />
        <span className={styles.driftSq} style={{ animationDelay: '2.5s', width: 14, height: 14, opacity: 0.35 }} />
        <span className={styles.driftSqOutline} style={{ animationDelay: '5s', width: 18, height: 18, opacity: 0.5 }} />
      </div>

      <div className={`${styles.driftCluster} ${styles.clusterRight}`}>
        <span className={styles.driftSqOutline} style={{ animationDelay: '1.5s', width: 24, height: 24, opacity: 0.4 }} />
        <span className={styles.driftSq} style={{ animationDelay: '4s', width: 16, height: 16, opacity: 0.55 }} />
        <span className={styles.driftSq} style={{ animationDelay: '6.5s', width: 10, height: 10, opacity: 0.3 }} />
      </div>

      {/* Ambient Grid Accent */}
      <div className={styles.gridAccent} />
    </div>
  );
}
