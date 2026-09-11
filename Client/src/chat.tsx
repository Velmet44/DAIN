import { useCallback, useEffect, useRef, useState } from "react";
import {
  defaultMaxTokens,
  defaultModel,
  listModels,
  streamCompletion,
  type ChatMessage,
  type SseFrame,
} from "./api";
import { MessageBubble, useLocalStorage } from "./components";
import { logDebug, logError, logInfo, logOk, logWarn } from "./logs";

interface Props {
  baseUrl: string;
  apiKey: string;
  setBaseUrl: (v: string) => void;
  setApiKey: (v: string) => void;
}

/** Streaming flushes are coalesced to this cadence so the growing transcript is
 * not re-rendered (and re-marked-down) on every token. */
const STREAM_FLUSH_MS = 60;

export function ChatView({ baseUrl, apiKey, setBaseUrl, setApiKey }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [modelId, setModelId] = useLocalStorage("dain:model", defaultModel());
  const [maxTokens, setMaxTokens] = useState<number>(defaultMaxTokens());
  const [models, setModels] = useState<string[]>([]);
  const [status, setStatus] = useState("");
  const [streaming, setStreaming] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  // Abort any in-flight completion when the view unmounts — the coordinator's
  // SSE generator would otherwise keep pumping frames into a dead component.
  useEffect(() => {
    return () => controllerRef.current?.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    listModels(baseUrl, apiKey, controller.signal)
      .then((models) => {
        logOk(`model picker: ${models.join(", ") || "(none)"}`);
        setModels(models);
        if (models.length > 0 && !modelId) {
          logInfo(`auto-selecting first model: ${models[0]}`);
          setModelId(models[0]);
        }
      })
      .catch((err: Error) => {
        if (controller.signal.aborted) return; // unmounted / superseded
        logError(`models unavailable: ${err.message}`);
        setStatus(`models: ${err.message}`);
      });
    return () => controller.abort();
  }, [baseUrl, apiKey, modelId]);

  const send = async (prompt?: string) => {
    const text = (prompt ?? input).trim();
    if (!text || streaming || !modelId) return;
    // Prompt content is debug-only: it must never reach the console in a
    // production build.
    logInfo(`send model=${modelId} promptChars=${text.length} maxTokens=${maxTokens}`);
    logDebug(`prompt: ${text}`);
    setInput("");
    setMessages((m) => [
      ...m,
      { role: "user", content: text },
      { role: "assistant", content: "", streaming: true },
    ]);
    setStreaming(true);
    setStatus("");
    const controller = new AbortController();
    controllerRef.current = controller;
    let acc = "";
    let committed = "";
    const flush = () => {
      if (acc === committed) return;
      committed = acc;
      setMessages((m) => {
        const tail = [...m];
        tail[tail.length - 1] = { role: "assistant", content: acc, streaming: true };
        return tail;
      });
    };
    const flusher = window.setInterval(flush, STREAM_FLUSH_MS);
    try {
      await streamCompletion(
        baseUrl,
        apiKey,
        { modelId, prompt: text, maxTokens, signal: controller.signal },
        (frame: SseFrame) => {
          if (frame.token) {
            acc += frame.token;
            logDebug(
              `token frame: "${frame.token.length > 40 ? `${frame.token.slice(0, 40)}…` : frame.token}" (total ${acc.length} chars)`,
            );
          }
          if (frame.type === "final" || frame.type === "error") {
            if (frame.type === "error") logError(`stream error frame: ${frame.detail}`);
            setStatus(`done (${frame.usage?.tokens ?? acc.length} tokens)`);
          }
        },
      );
      flush();
      setMessages((m) => {
        const tail = [...m];
        tail[tail.length - 1] = { role: "assistant", content: acc, streaming: false };
        return tail;
      });
    } catch (err) {
      const aborted = controller.signal.aborted;
      if (aborted) logWarn(`completion aborted, ${acc.length} chars received`);
      else logError(`completion failed: ${(err as Error).message}`);
      setStatus(aborted ? "stopped" : `error: ${(err as Error).message}`);
      setMessages((m) => {
        const tail = [...m];
        tail[tail.length - 1] = {
          role: "assistant",
          content: aborted ? acc || "⚠ stopped" : `⚠ ${(err as Error).message}`,
          streaming: false,
        };
        return tail;
      });
    } finally {
      window.clearInterval(flusher);
      setStreaming(false);
      controllerRef.current = null;
    }
  };

  const stop = useCallback(() => {
    logInfo("stop requested");
    controllerRef.current?.abort();
  }, []);

  return (
    <div className="view">
      <div className="toolbar">
        <input
          placeholder="coordinator URL"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
        />
        <input
          placeholder="API key"
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
        />
        <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
          {models.length === 0 && <option value="">no models (check URL/key)</option>}
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        <input
          className="narrow"
          type="number"
          min={1}
          max={512}
          value={maxTokens}
          onChange={(e) => setMaxTokens(Number(e.target.value))}
          title="max tokens"
        />
        <span className="status">{status}</span>
      </div>
      <div className="thread">
        {messages.length === 0 && (
          <p className="hint">
            Prototype model {modelId || "?"} — output is seeded proto-text, not language. Say hi.
          </p>
        )}
        {messages.map((m, i) => (
          <MessageBubble key={i} message={m} onStop={stop} />
        ))}
      </div>
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <input
          placeholder="Type a prompt and press Enter"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={streaming}
        />
        <button type="submit" disabled={streaming || !input.trim()}>
          {streaming ? "streaming…" : "Send"}
        </button>
        {streaming && (
          <button type="button" onClick={stop}>
            Stop
          </button>
        )}
      </form>
    </div>
  );
}