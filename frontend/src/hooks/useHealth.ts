import { useState, useEffect } from 'react';
import { fetchHealth } from '../api/client';
import type { HealthResponse } from '../api/types';

export interface UseHealthResult {
  health: HealthResponse | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

export function useHealth(pollIntervalMs = 30_000): UseHealthResult {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const refresh = () => setTick(t => t + 1);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchHealth()
      .then(h => { if (!cancelled) { setHealth(h); setError(null); } })
      .catch(() => { if (!cancelled) setError('Backend unreachable'); })
      .finally(() => { if (!cancelled) setLoading(false); });

    const id = setInterval(refresh, pollIntervalMs);
    return () => { cancelled = true; clearInterval(id); };
  }, [tick, pollIntervalMs]);

  return { health, loading, error, refresh };
}
