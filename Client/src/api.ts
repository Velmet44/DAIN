export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

export interface CompletionRequest {
  modelId: string;
  prompt: string;
  maxTokens: number;
  signal?: AbortSignal;
}

export interface SseFrame {
  type?: string;
  job_id?: string;
  token?: string;
  finish_reason?: string;
  usage?: { tokens?: number };
  detail?: string;
}

export interface NodeRow {
  node_id: string;
  state: string;
  score: number;
  score_components?: Record<string, number>;
  connected: boolean;
  gpu?: string;
  vram_free_gb?: number;
  agent_version?: string;
  uptime_ratio?: number;
  failure_rate?: number;
}

export interface PlacementEvent {
  ts: number;
  trigger: string;
  model_id: string;
  k?: number;
  degraded: boolean;
  node_ids: string[];
}

export interface ClusterStatus {
  nodes: NodeRow[];
  placements: PlacementEvent[];
  active_jobs: number;
}

const DEFAULT_MAX_TOKENS = import.meta.env.VITE_DEFAULT_MAX_TOKENS || "64";

export function defaultApiUrl(): string {
  return import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
}

export function defaultApiKey(): string {
  return import.meta.env.VITE_API_KEY || "";
}

export function defaultModel(): string {
  return import.meta.env.VITE_MODEL_ID || "";
}

export function defaultMaxTokens(): number {
  return Number.parseInt(DEFAULT_MAX_TOKENS, 10) || 64;
}

export async function listModels(baseUrl: string, apiKey: string): Promise<string[]> {
  const resp = await fetch(`${baseUrl}/v1/models`, { headers: { "X-API-Key": apiKey } });
  if (!resp.ok) throw new Error(`GET /v1/models -> ${resp.status}`);
  const body = (await resp.json()) as { models: { model_id: string }[] };
  return body.models.map((m) => m.model_id);
}

export async function clusterStatus(baseUrl: string, apiKey: string): Promise<ClusterStatus> {
  const resp = await fetch(`${baseUrl}/v1/nodes`, { headers: { "X-API-Key": apiKey } });
  if (!resp.ok) throw new Error(`GET /v1/nodes -> ${resp.status}`);
  return (await resp.json()) as ClusterStatus;
}

/** POST a streaming completion and feed SSE frames to `onFrame` until done. */
export async function streamCompletion(
  baseUrl: string,
  apiKey: string,
  req: CompletionRequest,
  onFrame: (frame: SseFrame) => void,
): Promise<void> {
  const resp = await fetch(`${baseUrl}/v1/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      "X-API-Key": apiKey,
    },
    body: JSON.stringify({
      model_id: req.modelId,
      prompt: req.prompt,
      max_tokens: req.maxTokens,
      stream: true,
    }),
    signal: req.signal,
  });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = (await resp.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* empty body */
    }
    throw new Error(detail);
  }
  const reader = resp.body?.getReader();
  if (!reader) throw new Error("no response body");
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice("data: ".length).trim();
      if (payload === "[DONE]") return;
      try {
        onFrame(JSON.parse(payload) as SseFrame);
      } catch {
        /* ignore malformed frame */
      }
    }
  }
}