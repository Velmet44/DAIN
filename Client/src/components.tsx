import { memo, useCallback, useMemo, useState } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";
import type { ChatMessage } from "./api";
import { logDebug, logInfo, logWarn } from "./logs";

function renderMarkdown(content: string): string {
  // Model/user content is untrusted: sanitize the rendered markdown before it
  // touches the DOM (no sanitizer here would be an XSS vector).
  return DOMPurify.sanitize(marked.parse(content || "…", { async: false }) as string);
}

/** Memoized bubble: unchanged messages skip re-rendering entirely, and each
 * bubble's markdown is parsed+cleaned only when its own content changes — the
 * whole transcript is no longer re-parsed on every streamed token. */
export const MessageBubble = memo(function MessageBubble({
  message,
  onStop,
}: {
  message: ChatMessage;
  onStop?: () => void;
}) {
  const html = useMemo(() => renderMarkdown(message.content), [message.content]);
  return (
    <div className={`bubble ${message.role}`}>
      {message.role === "assistant" && message.streaming && (
        <button className="stop" onClick={onStop} title="stop">
          ■
        </button>
      )}
      <div dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
});

export function useLocalStorage(key: string, initial: string): [string, (v: string) => void] {
  // Only persist values the user explicitly edits. On every fresh page load the
  // env-provided default wins, so a stale value from a previous session (old
  // coordinator port / API key) never overrides the launcher's current config.
  const [value, setValue] = useState<string>(() => {
    try {
      const raw = localStorage.getItem(key);
      if (raw) {
        const parsed = JSON.parse(raw) as [string, boolean];
        if (Array.isArray(parsed) && parsed[1]) {
          logDebug(`${key}: using user-edited value`);
          return parsed[0];
        }
        logWarn(`${key}: ignoring stale local (not user-edited) value; using env default`);
      }
    } catch {
      logWarn(`${key}: ignoring legacy local value; using env default`);
    }
    return initial;
  });

  const setPersisted = useCallback(
    (v: string) => {
      setValue(v);
      logInfo(`${key}: persisted (user-edited)`);
      try {
        localStorage.setItem(key, JSON.stringify([v, true]));
      } catch {
        /* storage unavailable */
      }
    },
    [key],
  );

  return [value, setPersisted];
}