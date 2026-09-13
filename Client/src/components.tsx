import { memo, useMemo, useState } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";
import type { ChatMessage, MsgStats } from "./types";
import { logDebug } from "./logs";

function renderMarkdown(content: string): string {
  // Model/user content is untrusted: sanitize the rendered markdown before it
  // touches the DOM (no sanitizer here would be an XSS vector).
  return DOMPurify.sanitize(marked.parse(content || "…", { async: false }) as string);
}

function fmtMs(ms: number | null): string {
  if (ms == null) return "–";
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

/** Decode throughput: tokens over the streaming window (after first token),
 * falling back to the whole request duration. */
function tokensPerSecond(stats: MsgStats): string {
  const totalS = stats.totalMs / 1000;
  if (stats.tokens <= 0 || totalS <= 0) return "–";
  const genS =
    stats.ttftMs != null && stats.totalMs > stats.ttftMs
      ? (stats.totalMs - stats.ttftMs) / 1000
      : totalS;
  return `${(Math.max(stats.tokens - 1, 1) / Math.max(genS, 1e-3)).toFixed(1)} tok/s`;
}

function StatsFooter({ stats }: { stats: MsgStats }) {
  const parts = [
    stats.model,
    tokensPerSecond(stats),
    `first token ${fmtMs(stats.ttftMs)}`,
    `total ${fmtMs(stats.totalMs)}`,
    `${stats.tokens} tok`,
  ];
  if (stats.stopped) parts.push("stopped");
  if (stats.error) parts.push("error");
  return <div className="msg-stats mono">{parts.join("  ·  ")}</div>;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="msg-copy"
      type="button"
      title="Copy message"
      onClick={() => {
        navigator.clipboard
          ?.writeText(text)
          .then(() => {
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1200);
          })
          .catch(() => setCopied(false));
      }}
    >
      {copied ? "copied" : "copy"}
    </button>
  );
}

/** One transcript message. Memoized: unchanged messages (incl. their parsed
 * markdown) skip re-rendering entirely while a sibling streams. */
export const MessageBubble = memo(function MessageBubble({
  message,
  onStop,
}: {
  message: ChatMessage;
  onStop?: () => void;
}) {
  const isUser = message.role === "user";
  const html = useMemo(
    () => (isUser ? null : renderMarkdown(message.content)),
    [message.content, isUser],
  );
  logDebug(`render bubble ${message.id} (${message.role}, ${message.content.length} chars)`);
  return (
    <div className={`row ${isUser ? "user" : "assistant"}`}>
      <div className={`bubble ${isUser ? "user" : "assistant"}`}>
        {isUser ? (
          <div className="plain">{message.content}</div>
        ) : (
          <>
            <div className="md" dangerouslySetInnerHTML={{ __html: html || "" }} />
            {message.streaming && <span className="cursor" aria-hidden />}
          </>
        )}
      </div>
      {!isUser && !message.streaming && (
        <div className="msg-meta">
          {message.stats && <StatsFooter stats={message.stats} />}
          {message.content && <CopyButton text={message.content} />}
        </div>
      )}
      {message.streaming && onStop && (
        <button className="stop" type="button" onClick={onStop} title="Stop generating">
          ■ stop
        </button>
      )}
    </div>
  );
});
