import { useEffect, useState } from "react";
import { marked } from "marked";
import type { ChatMessage } from "./api";

export function MessageBubble({
  message,
  onStop,
}: {
  message: ChatMessage;
  onStop?: () => void;
}) {
  const html = marked.parse(message.content || "…", { async: false }) as string;
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
}

export function useLocalStorage(key: string, initial: string): [string, (v: string) => void] {
  const [value, setValue] = useState<string>(() => localStorage.getItem(key) ?? initial);
  useEffect(() => {
    localStorage.setItem(key, value);
  }, [key, value]);
  return [value, setValue];
}