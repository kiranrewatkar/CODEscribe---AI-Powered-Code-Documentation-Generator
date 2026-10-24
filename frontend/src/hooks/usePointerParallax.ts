import { useEffect, useRef } from 'react';

interface PointerParallaxOptions {
  disabled?: boolean;
}

export function usePointerParallax<T extends HTMLElement = HTMLDivElement>(
  options: PointerParallaxOptions = {}
) {
  const ref = useRef<T>(null);
  const { disabled = false } = options;

  useEffect(() => {
    const el = ref.current;
    if (!el || disabled) return;

    // Check for reduced motion or coarse pointer (touch devices)
    if (window.matchMedia('(prefers-reduced-motion: reduce), (pointer: coarse)').matches) {
      return;
    }

    let rafId: number | null = null;
    let targetX = 0;
    let targetY = 0;

    const onPointerMove = (e: PointerEvent) => {
      const rect = el.getBoundingClientRect();
      const normX = (e.clientX - rect.left) / rect.width - 0.5;
      const normY = (e.clientY - rect.top) / rect.height - 0.5;

      targetX = Math.max(-0.5, Math.min(0.5, normX));
      targetY = Math.max(-0.5, Math.min(0.5, normY));

      if (rafId === null) {
        rafId = requestAnimationFrame(() => {
          el.style.setProperty('--mouse-x', targetX.toFixed(4));
          el.style.setProperty('--mouse-y', targetY.toFixed(4));
          rafId = null;
        });
      }
    };

    const onPointerLeave = () => {
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      el.style.setProperty('--mouse-x', '0');
      el.style.setProperty('--mouse-y', '0');
    };

    el.addEventListener('pointermove', onPointerMove, { passive: true });
    el.addEventListener('pointerleave', onPointerLeave, { passive: true });

    return () => {
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
      }
      el.removeEventListener('pointermove', onPointerMove);
      el.removeEventListener('pointerleave', onPointerLeave);
    };
  }, [disabled]);

  return ref;
}
