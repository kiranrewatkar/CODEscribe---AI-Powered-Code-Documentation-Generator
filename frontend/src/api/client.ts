/**
 * API client for the CODEscribe local inference backend.
 * All requests go through the Vite proxy at /api → http://127.0.0.1:8000
 */

import type {
  HealthResponse,
  ExamplesResponse,
  DocumentRequest,
  DocumentResponse,
  ApiError,
} from './types';

const BASE = '';  // Vite proxy handles /api → 127.0.0.1:8000

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // ignore parse errors
    }
    const err: ApiError = { detail, status: res.status };
    throw err;
  }

  return res.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/api/health');
}

export async function fetchExamples(): Promise<ExamplesResponse> {
  return request<ExamplesResponse>('/api/examples');
}

export async function generateDocument(payload: DocumentRequest): Promise<DocumentResponse> {
  return request<DocumentResponse>('/api/document', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}
