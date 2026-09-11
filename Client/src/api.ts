import { logDebug, logError, logInfo, logOk } from "./logs";

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
  status?: string;
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

/** Upper bound for plain fetches (model list, dashboard poll): a hung
 * coordinator must never block the UI forever. */
const FETCH_TIMEOUT_MS = 15000;

/** A streaming completion that stops producing frames for this long (no HTTP
 * activity at all) is presumed dead and aborted. */
const STREAM_IDLE_TIMEOUT_MS = 30000;

/** fetch() that aborts on an optional external signal and on a wall-clock
 * timeout. Timeouts surface as `Error("request timed out ...")` so callers can
 * tell them apart from a user-initiated abort. */
function fetchWithTimeout(
  url: string,
  init: RequestInit = {},
  timeoutMs: number = FETCH_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  const external = init.signal;
  let timedOut = false;
  const onAbort = () => controller.abort();
  if (external) {
    if (external.aborted) controller.abort();
    else external.addEventListener("abort", onAbort);
  }
  const timer = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  return fetch(url, { ...init, signal: controller.signal })
    .catch((err: unknown) => {
      if (timedOut) throw new Error(`request timed out after ${timeoutMs}ms`);
      throw err;
    })
    .finally(() => {
      window.clearTimeout(timer);
      if (external) external.removeEventListener("abort", onAbort);
    });
}

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

export async function listModels(
  baseUrl: string,
  apiKey: string,
  signal?: AbortSignal,
): Promise<string[]> {
  logInfo(`GET ${baseUrl}/v1/models (key ${apiKey ? "set" : "EMPTY"})`);
  const resp = await fetchWithTimeout(`${baseUrl}/v1/models`, {
    headers: { "X-API-Key": apiKey },
    signal,
  });
  if (!resp.ok) {
    logError(`GET /v1/models -> ${resp.status}`);
    throw new Error(`GET /v1/models -> ${resp.status}`);
  }
  const body = (await resp.json()) as { models: { model_id: string }[] };
  logOk(`models: ${body.models.length} available`);
  return body.models.map((m) => m.model_id);
}

export async function clusterStatus(
  baseUrl: string,
  apiKey: string,
  signal?: AbortSignal,
): Promise<ClusterStatus> {
  logDebug(`GET ${baseUrl}/v1/nodes (key ${apiKey ? "set" : "EMPTY"})`);
  const resp = await fetchWithTimeout(`${baseUrl}/v1/nodes`, {
    headers: { "X-API-Key": apiKey },
    signal,
  });
  if (!resp.ok) {
    logError(`GET /v1/nodes -> ${resp.status}`);
    throw new Error(`GET /v1/nodes -> ${resp.status}`);
  }
  return (await resp.json()) as ClusterStatus;
}

/** POST a streaming completion and feed SSE frames to `onFrame` until done.
 *
 * The caller's `req.signal` (Stop / unmount) aborts the fetch; additionally an
 * idle watchdog aborts if no HTTP data arrives within `STREAM_IDLE_TIMEOUT_MS`,
 * so a coordinator that silently stops streaming can't wedge the UI forever. */
export async function streamCompletion(
  baseUrl: string,
  apiKey: string,
  req: CompletionRequest,
  onFrame: (frame: SseFrame) => void,
): Promise<void> {
  logInfo(
    `POST ${baseUrl}/v1/completions model=${req.modelId} promptLen=${req.prompt.length} maxTokens=${req.maxTokens}`,
  );
  const controller = new AbortController();
  const external = req.signal;
  let timedOut = false;
  let idleTimer: number | undefined;
  const armIdle = () => {
    window.clearTimeout(idleTimer);
    idleTimer = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, STREAM_IDLE_TIMEOUT_MS);
  };
  const disarmIdle = () => window.clearTimeout(idleTimer);
  const onAbort = () => controller.abort();
  if (external) {
    if (external.aborted) controller.abort();
    else external.addEventListener("abort", onAbort);
  }
  try {
    armIdle();
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
      signal: controller.signal,
    });
    disarmIdle();
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try {
        const body = (await resp.json()) as { detail?: string };
        if (body.detail) detail = body.detail;
      } catch {
        /* empty body */
      }
      logError(`POST /v1/completions -> ${detail}`);
      throw new Error(detail);
    }
    const reader = resp.body?.getReader();
    if (!reader) throw new Error("no response body");
    const decoder = new TextDecoder();
    let buffer = "";
    let frames = 0;
    let tokens = 0;
    for (;;) {
      armIdle();
      const { done, value } = await reader.read();
      disarmIdle();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice("data: ".length).trim();
        if (payload === "[DONE]") {
          logOk(`stream complete: ${frames} frames, ${tokens} tokens`);
          return;
        }
        try {
          const frame = JSON.parse(payload) as SseFrame;
          frames += 1;
          if (frame.token) tokens += 1;
          if (frame.job_id && frame.status) logOk(`job ${frame.job_id} ${frame.status}`);
          if (frame.type === "error") logError(`job ${frame.job_id ?? "?"} error: ${frame.detail}`);
          if (frame.type === "final")
            logOk(`final finish=${frame.finish_reason} usageTokens=${frame.usage?.tokens ?? tokens}`);
          onFrame(frame);
        } catch {
          /* ignore malformed frame */
        }
      }
    }
  } catch (err) {
    if (timedOut) throw new Error(`stream idle timeout (no data for ${STREAM_IDLE_TIMEOUT_MS}ms)`);
    throw err;
  } finally {
    disarmIdle();
    if (external) external.removeEventListener("abort", onAbort);
  }
}