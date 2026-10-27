// API type definitions matching backend schemas exactly.
// See app/server.py for the source of truth.

export interface HealthResponse {
  status: 'ok' | 'degraded';
  model_loaded: boolean;
  model_path: string;
  device: string;
  cuda_available: boolean;
  gpu_name: string | null;
  vram_allocated_mb: number | null;
  vram_total_mb: number | null;
  torch_version: string;
  transformers_version: string;
}

export interface ExampleSnippet {
  id: string;
  title: string;
  description: string;
  code: string;
}

export interface ExamplesResponse {
  examples: ExampleSnippet[];
}

export interface DocumentRequest {
  code: string;
  num_beams?: number;
  max_new_tokens?: number;
  temperature?: number;
  do_sample?: boolean;
  include_mermaid?: boolean;
}

export interface ASTMetadata {
  function_names: string[];
  parameters: string[];
  return_annotation: string | null;
  docstring_stripped: boolean;
  syntax_valid: boolean;
}

export interface TimingInfo {
  preprocessing_ms: number;
  inference_ms: number;
  mermaid_ms: number;
  total_ms: number;
}

export interface DocumentResponse {
  success: boolean;
  markdown: string;
  docstring: string;
  mermaid_code: string | null;
  ast_metadata: ASTMetadata;
  timing_ms: TimingInfo;
  device: string;
  status: string;
  error_message: string | null;
}

export interface ApiError {
  detail: string;
  status: number;
}

export interface OptimizationFinding {
  rule_id: string;
  title: string;
  description: string;
  suggestion: string;
  line_number: number;
  column_offset: number;
  function_scope: string | null;
}

export interface OptimizeRequest {
  code: string;
}

export interface OptimizeResponse {
  success: boolean;
  findings: OptimizationFinding[];
  error_message: string | null;
  disclaimer: string;
  total_findings: number;
}
