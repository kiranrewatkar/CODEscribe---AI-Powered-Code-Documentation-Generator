import { useEffect, useRef } from 'react';
import styles from './InteractiveParticles.module.css';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  shape: 'square' | 'circle';
  rotation: number;
  rotationSpeed: number;
  color: string;
  baseAlpha: number;
  alpha: number;
}

interface InteractiveParticlesProps {
  className?: string;
  particleCount?: number;
  interactionRadius?: number;
  themeMode?: 'coral' | 'cream' | 'auto';
}

export function InteractiveParticles({
  className = '',
  particleCount = 48,
  interactionRadius = 130,
  themeMode = 'coral',
}: InteractiveParticlesProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Accessibility check: disable continuous motion if user prefers reduced motion
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let width = canvas.offsetWidth || window.innerWidth;
    let height = canvas.offsetHeight || window.innerHeight;

    const updateDimensions = () => {
      if (!canvas || !ctx) return;
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const displayWidth = Math.floor(rect.width || canvas.offsetWidth || window.innerWidth);
      const displayHeight = Math.floor(rect.height || canvas.offsetHeight || window.innerHeight);

      if (displayWidth > 0 && displayHeight > 0) {
        width = displayWidth;
        height = displayHeight;
        canvas.width = Math.floor(displayWidth * dpr);
        canvas.height = Math.floor(displayHeight * dpr);
        if (typeof ctx.resetTransform === 'function') {
          ctx.resetTransform();
        } else {
          ctx.setTransform(1, 0, 0, 1, 0, 0);
        }
        ctx.scale(dpr, dpr);
      }
    };

    updateDimensions();

    // Mouse coordinates (null when outside canvas/window)
    const mouse = { x: -9999, y: -9999, active: false };

    // Responsive particle count adjustment
    const effectiveCount = window.innerWidth < 768 ? Math.floor(particleCount * 0.5) : particleCount;

    // Palette harmonized with CODEscribe Symboolic Editorial theme
    const colors =
      themeMode === 'coral'
        ? [
            '255, 254, 252', // Warm White
            '241, 213, 209', // Coral Soft
            '230, 141, 145', // Coral
            '18, 18, 16',     // Ink / Charcoal
          ]
        : [
            '230, 141, 145', // Coral
            '212, 116, 120', // Coral Dark
            '18, 18, 16',     // Charcoal
            '87, 83, 79',     // Muted Body
          ];

    // Initialize particles
    const particles: Particle[] = [];
    for (let i = 0; i < effectiveCount; i++) {
      const isSquare = i % 3 !== 0; // 66% geometric squares (mosaic theme), 33% soft circular nodes
      const color = colors[Math.floor(Math.random() * colors.length)];
      const baseAlpha = color === '18, 18, 16' ? 0.22 : 0.42;

      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.6,
        vy: (Math.random() - 0.5) * 0.6,
        size: isSquare ? Math.random() * 7 + 4 : Math.random() * 3 + 2,
        shape: isSquare ? 'square' : 'circle',
        rotation: Math.random() * Math.PI * 2,
        rotationSpeed: (Math.random() - 0.5) * 0.012,
        color,
        baseAlpha,
        alpha: baseAlpha,
      });
    }

    // Pointer move listener on window (passive, zero impact on clicks/scrolling)
    const handlePointerMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const clientX = e.clientX;
      const clientY = e.clientY;

      if (
        clientX >= rect.left &&
        clientX <= rect.right &&
        clientY >= rect.top &&
        clientY <= rect.bottom
      ) {
        mouse.x = clientX - rect.left;
        mouse.y = clientY - rect.top;
        mouse.active = true;
      } else {
        mouse.active = false;
      }
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (e.touches.length > 0) {
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        if (
          touch.clientX >= rect.left &&
          touch.clientX <= rect.right &&
          touch.clientY >= rect.top &&
          touch.clientY <= rect.bottom
        ) {
          mouse.x = touch.clientX - rect.left;
          mouse.y = touch.clientY - rect.top;
          mouse.active = true;
        }
      }
    };

    const handlePointerLeave = () => {
      mouse.active = false;
      mouse.x = -9999;
      mouse.y = -9999;
    };

    window.addEventListener('mousemove', handlePointerMove, { passive: true });
    window.addEventListener('touchmove', handleTouchMove, { passive: true });
    window.addEventListener('touchend', handlePointerLeave, { passive: true });
    document.addEventListener('mouseleave', handlePointerLeave);
    window.addEventListener('resize', updateDimensions, { passive: true });

    // Render loop
    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Draw and update each particle
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];

        // Cursor interaction physics
        if (mouse.active) {
          const dx = p.x - mouse.x;
          const dy = p.y - mouse.y;
          const dist = Math.hypot(dx, dy);

          if (dist < interactionRadius && dist > 0) {
            // Smooth repulsion force pushing gently away from cursor
            const force = (1 - dist / interactionRadius) * 2.4;
            const angle = Math.atan2(dy, dx);
            p.x += Math.cos(angle) * force;
            p.y += Math.sin(angle) * force;

            // Highlight particle opacity on contact
            p.alpha = Math.min(0.9, p.baseAlpha + 0.38);
          } else {
            p.alpha += (p.baseAlpha - p.alpha) * 0.05;
          }
        } else {
          p.alpha += (p.baseAlpha - p.alpha) * 0.05;
        }

        // Standard organic movement
        p.x += p.vx;
        p.y += p.vy;
        p.rotation += p.rotationSpeed;

        // Wrap around boundaries smoothly
        if (p.x < -20) p.x = width + 20;
        else if (p.x > width + 20) p.x = -20;
        if (p.y < -20) p.y = height + 20;
        else if (p.y > height + 20) p.y = -20;

        // Draw particle
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rotation);
        ctx.fillStyle = `rgba(${p.color}, ${p.alpha.toFixed(2)})`;

        if (p.shape === 'square') {
          // Rounded geometric square matching mosaic theme
          const half = p.size / 2;
          ctx.beginPath();
          if (typeof ctx.roundRect === 'function') {
            ctx.roundRect(-half, -half, p.size, p.size, 2);
          } else {
            ctx.rect(-half, -half, p.size, p.size);
          }
          ctx.fill();
        } else {
          // Circular particle
          ctx.beginPath();
          ctx.arc(0, 0, p.size / 2, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();

        // Connect nearby particles with subtle editorial links
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const distBetween = Math.hypot(p.x - p2.x, p.y - p2.y);

          if (distBetween < 80) {
            const linkAlpha = (1 - distBetween / 80) * 0.12;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = `rgba(255, 254, 252, ${linkAlpha.toFixed(3)})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('mousemove', handlePointerMove);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handlePointerLeave);
      document.removeEventListener('mouseleave', handlePointerLeave);
      window.removeEventListener('resize', updateDimensions);
    };
  }, [particleCount, interactionRadius, themeMode]);

  return (
    <canvas
      ref={canvasRef}
      className={`${styles.particleCanvas} ${className}`}
      aria-hidden="true"
    />
  );
}
