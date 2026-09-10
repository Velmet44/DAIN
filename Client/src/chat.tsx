import { useMemo, useState } from "react";
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

export function ChatView({ baseUrl, apiKey, setBaseUrl, setApiKey }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [modelId, setModelId] = useLocalStorage("dain:model", defaultModel());
  const [maxTokens, setMaxTokens] = useState<number>(defaultMaxTokens());
  const [models, setModels] = useState<string[]>([]);
  const [status, setStatus] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [signal, setSignal] = useState<AbortController | null>(null);

  useMemo(() => {
    listModels(baseUrl, apiKey)
      .then((models) => {
        logOk(`model picker: ${models.join(", ") || "(none)"}`);
        setModels(models);
      })
      .catch((err: Error) => {
        logError(`models unavailable: ${err.message}`);
        setStatus(`models: ${err.message}`);
      });
  }, [baseUrl, apiKey]);

  const send = async (prompt?: string) => {
    const text = (prompt ?? input).trim();
    if (!text || streaming || !modelId) return;
    logInfo(`send model=${modelId} prompt="${text.slice(0, 80)}${text.length > 80 ? "…" : ""}" maxTokens=${maxTokens}`);
    setInput("");
    setMessages((m) => [
      ...m,
      { role: "user", content: text },
      { role: "assistant", content: "", streaming: true },
    ]);
    setStreaming(true);
    setStatus("");
    const controller = new AbortController();
    setSignal(controller);
    let acc = "";
    try {
      await streamCompletion(
        baseUrl,
        apiKey,
        { modelId, prompt: text, maxTokens, signal: controller.signal },
        (frame: SseFrame) => {
          if (frame.token) {
            acc += frame.token;
            logDebug(`token frame: "${frame.token.length > 40 ? `${frame.token.slice(0, 40)}…` : frame.token}" (total ${acc.length} chars)`);
            setMessages((m) => {
              const tail = [...m];
              tail[tail.length - 1] = { role: "assistant", content: acc, streaming: true };
              return tail;
            });
          }
          if (frame.type === "final" || frame.type === "error") {
            if (frame.type === "error") logError(`stream error frame: ${frame.detail}`);
            setStatus(`done (${frame.usage?.tokens ?? acc.length} tokens)`);
          }
        },
      );
      setMessages((m) => {
        const tail = [...m];
        tail[tail.length - 1] = { role: "assistant", content: acc, streaming: false };
        return tail;
      });
    } catch (err) {
      const aborted = controller.signal.aborted;
      if (aborted) logWarn(`completion aborted, ${acc.length} chars received`);
      else logError(`completion failed: ${(err as Error).message}`);
      setStatus(`error: ${(err as Error).message}`);
      setMessages((m) => {
        const tail = [...m];
        tail[tail.length - 1] = {
          role: "assistant",
          content: `⚠ ${(err as Error).message}`,
          streaming: false,
        };
        return tail;
      });
    } finally {
      setStreaming(false);
      setSignal(null);
    }
  };

  const stop = () => {
    logInfo("stop requested");
    signal?.abort();
  };

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