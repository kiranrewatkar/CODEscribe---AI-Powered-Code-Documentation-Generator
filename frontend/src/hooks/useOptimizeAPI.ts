import { useState, useCallback } from 'react';
import { analyzeOptimizations } from '../api/client';
import type { OptimizeResponse, ApiError } from '../api/types';

export function useOptimizeAPI() {
  const [data, setData] = useState<OptimizeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async (code: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeOptimizations({ code });
      setData(res);
      return res;
    } catch (err) {
      const apiErr = err as ApiError;
      const msg = apiErr.detail ?? 'Failed to analyze code optimizations.';
      setError(msg);
      setData(null);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
  }, []);

  return { analyze, data, loading, error, reset };
}
