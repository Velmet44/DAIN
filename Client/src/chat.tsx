import { useCallback, useEffect, useRef, useState } from "react";
import { listModels, streamCompletion } from "./api";
import { MessageBubble } from "./components";
import type { ChatMessage, ClientSettings, Conversation, MsgStats } from "./types";
import { titleFromMessage } from "./storage";
import { logDebug, logError, logInfo, logOk, logWarn } from "./logs";

interface Props {
  conv: Conversation;
  settings: ClientSettings;
  onModelPicked: (modelId: string) => void;
  onPatch: (patch: {
    title?: string;
    appendMessage?: ChatMessage;
    replaceMessage?: ChatMessage;
  }) => void;
  onConnected: (ok: boolean) => void;
}

const EXAMPLES = [
  "Explain pipeline-parallel inference like I'm five",
  "Write a Python function that merges overlapping intervals",
  "What are the trade-offs of int4 quantization?",
  "Summarize how a raft consensus cluster handles a node failure",
];

/** Streaming flushes are coalesced so the growing transcript is not re-rendered
 * (and re-marked-down) on every token. */
const STREAM_FLUSH_MS = 60;

/**
 * Compose the raw-completion prompt the coordinator expects: optional system
 * preamble + a User/Assistant transcript. With history off, only the latest
 * user message is included.
 */
export function buildPrompt(
  messages: ChatMessage[],
  systemPrompt: string,
  sendHistory: boolean,
): string {
  const usable = messages.filter((m) => m.content && !m.stats?.error);
  const turns = sendHistory ? usable : usable.slice(-1);
  const lines: string[] = [];
  if (systemPrompt.trim()) lines.push(systemPrompt.trim());
  for (const m of turns) {
    lines.push(m.role === "user" ? `User: ${m.content}` : `Assistant: ${m.content}`);
  }
  lines.push("Assistant:");
  return lines.join("\n\n");
}

export function ChatView({ conv, settings, onModelPicked, onPatch, onConnected }: Props) {
  const [models, setModels] = useState<string[]>([]);
  const [modelError, setModelError] = useState("");
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const threadRef = useRef<HTMLDivElement | null>(null);

  const modelId = settings.modelId;

  // Model picker + connection indicator refresh when the endpoint changes.
  useEffect(() => {
    const controller = new AbortController();
    listModels(settings.url, settings.apiKey, controller.signal)
      .then((list) => {
        logOk(`model picker: ${list.join(", ") || "(none)"}`);
        setModels(list);
        setModelError("");
        onConnected(true);
        if (list.length > 0 && !list.includes(modelId)) {
          logInfo(`auto-selecting first model: ${list[0]}`);
          onModelPicked(list[0]);
        }
      })
      .catch((err: Error) => {
        if (controller.signal.aborted) return;
        logError(`models unavailable: ${err.message}`);
        setModelError(err.message);
        onConnected(false);
      });
    return () => controller.abort();
    // Model changes must not refetch; only endpoint changes do.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.url, settings.apiKey]);

  // Keep the newest message in view while it streams.
  useEffect(() => {
    const thread = threadRef.current;
    if (!thread) return;
    const nearBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 160;
    if (nearBottom) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conv.messages]);

  // Abort any in-flight completion when the view unmounts or the chat switches.
  useEffect(() => () => controllerRef.current?.abort(), [conv.id]);

  const stop = useCallback(() => {
    logInfo("stop requested");
    controllerRef.current?.abort();
  }, []);

  const send = async (text?: string) => {
    const prompt = (text ?? input).trim();
    if (!prompt || streaming) return;
    if (!modelId) {
      setModelError("no model selected — check Settings → connection");
      return;
    }
    logInfo(`send model=${modelId} promptChars=${prompt.length} maxTokens=${settings.maxTokens}`);
    logDebug(`prompt: ${prompt}`);
    setInput("");

    const stamp = Date.now().toString(36);
    const userMsg: ChatMessage = {
      id: `m-${stamp}-u`,
      role: "user",
      content: prompt,
      ts: Date.now(),
    };
    const assistantMsg: ChatMessage = {
      id: `m-${stamp}-a`,
      role: "assistant",
      content: "",
      streaming: true,
      ts: Date.now(),
    };
    onPatch({ appendMessage: userMsg });
    if (conv.title === "New chat" || !conv.title) {
      onPatch({ title: titleFromMessage(prompt) });
    }
    onPatch({ appendMessage: assistantMsg });

    // Prompt composition uses the transcript *including* the new user message.
    const composed = buildPrompt(
      [...conv.messages, userMsg],
      settings.systemPrompt,
      settings.sendHistory,
    );

    const controller = new AbortController();
    controllerRef.current = controller;
    const startedAt = performance.now();
    let firstTokenAt: number | null = null;
    let acc = "";
    let committed = "";
    let tokens = 0;
    const flush = () => {
      if (acc === committed) return;
      committed = acc;
      onPatch({
        replaceMessage: { ...assistantMsg, content: acc, streaming: true },
      });
    };
    const flusher = window.setInterval(flush, STREAM_FLUSH_MS);
    setStreaming(true);
    try {
      await streamCompletion(
        settings.url,
        settings.apiKey,
        { modelId, prompt: composed, maxTokens: settings.maxTokens, signal: controller.signal },
        (frame) => {
          if (frame.token) {
            if (firstTokenAt === null) firstTokenAt = performance.now();
            tokens += 1;
            acc += frame.token;
          }
          if (frame.type === "reset") {
            // Watchdog re-dispatched a replica pipeline: the transcript starts
            // over from the prompt, so drop everything accumulated so far.
            acc = "";
            committed = "";
            tokens = 0;
            flush();
            logWarn("watchdog reset: transcript regenerated from prompt");
          }
          if (frame.usage?.tokens != null) tokens = frame.usage.tokens;
        },
      );
      const totalMs = performance.now() - startedAt;
      const stats: MsgStats = {
        model: modelId,
        ttftMs: firstTokenAt === null ? null : Math.round(firstTokenAt - startedAt),
        totalMs: Math.round(totalMs),
        tokens,
      };
      logOk(
        `reply done: ${tokens} tok, ttft ${stats.ttftMs ?? "?"}ms, total ${Math.round(totalMs)}ms`,
      );
      onPatch({
        replaceMessage: { ...assistantMsg, content: acc, streaming: false, stats },
      });
    } catch (err) {
      const aborted = controller.signal.aborted;
      const message = (err as Error).message;
      if (aborted) logWarn(`completion aborted, ${acc.length} chars received`);
      else logError(`completion failed: ${message}`);
      const stats: MsgStats = {
        model: modelId,
        ttftMs: firstTokenAt === null ? null : Math.round(firstTokenAt - startedAt),
        totalMs: Math.round(performance.now() - startedAt),
        tokens,
        stopped: aborted,
        ...(aborted ? {} : { error: message }),
      };
      onPatch({
        replaceMessage: {
          ...assistantMsg,
          content: aborted ? acc : `⚠ ${message}`,
          streaming: false,
          stats,
        },
      });
    } finally {
      window.clearInterval(flusher);
      flush();
      setStreaming(false);
      controllerRef.current = null;
    }
  };

  const empty = conv.messages.length === 0;

  return (
    <div className="chat-view">
      <header className="chat-head">
        <select
          className="model-select"
          value={modelId}
          onChange={(e) => onModelPicked(e.target.value)}
          title="Model served by the coordinator"
        >
          {models.length === 0 && (
            <option value="">
              {modelError ? "unreachable — see Settings" : modelId || "no models"}
            </option>
          )}
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        <span className="chat-status">
          {streaming ? "generating…" : modelError ? `⚠ ${modelError}` : ""}
        </span>
      </header>

      <div className="thread" ref={threadRef}>
        {empty ? (
          <div className="welcome">
            <h1>DAIN</h1>
            <p>
              Distributed inference client{modelId ? ` — running ${modelId}` : ""}. Conversations
              are saved in this browser.
            </p>
            <div className="examples">
              {EXAMPLES.map((ex) => (
                <button key={ex} type="button" onClick={() => void send(ex)}>
                  {ex}
                </button>
              ))}
            </div>
          </div>
        ) : (
          conv.messages.map((m) => <MessageBubble key={m.id} message={m} onStop={stop} />)
        )}
        <div ref={bottomRef} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <input
          placeholder={
            modelId ? "Message the cluster…" : "Connect a coordinator in Settings first"
          }
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={streaming}
          autoFocus
        />
        {streaming ? (
          <button type="button" className="stop-btn" onClick={stop}>
            ■
          </button>
        ) : (
          <button type="submit" className="send-btn" disabled={!input.trim()}>
            ↑
          </button>
        )}
      </form>
    </div>
  );
}
