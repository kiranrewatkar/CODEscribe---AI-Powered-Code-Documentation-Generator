import { useState, useCallback, useRef } from 'react';
import { generateDocument } from '../api/client';
import type { DocumentRequest, DocumentResponse, ApiError } from '../api/types';

export interface UseDocumentAPIResult {
  generate: (payload: DocumentRequest) => Promise<void>;
  data: DocumentResponse | null;
  loading: boolean;
  error: string | null;
  reset: () => void;
}

export function useDocumentAPI(): UseDocumentAPIResult {
  const [data, setData] = useState<DocumentResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const generate = useCallback(async (payload: DocumentRequest) => {
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();

    setLoading(true);
    setError(null);
    setData(null);

    try {
      const result = await generateDocument(payload);
      setData(result);
    } catch (e: unknown) {
      const apiErr = e as ApiError;
      if (apiErr?.detail) {
        setError(apiErr.detail);
      } else if (e instanceof Error) {
        setError(e.message);
      } else {
        setError('Unexpected error occurred');
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
  }, []);

  return { generate, data, loading, error, reset };
}
