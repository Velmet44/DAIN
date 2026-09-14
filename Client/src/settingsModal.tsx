import { useEffect, useState } from "react";
import type { ClientSettings } from "./types";
import { DEFAULT_SYSTEM_PROMPT } from "./storage";
import { logInfo } from "./logs";

interface Props {
  settings: ClientSettings;
  onSave: (next: ClientSettings) => void;
  onClose: () => void;
}

/** Settings dialog: coordinator connection, generation defaults, and the
 * system prompt / history controls that shape what the model actually sees. */
export function SettingsModal({ settings, onSave, onClose }: Props) {
  const [draft, setDraft] = useState<ClientSettings>(settings);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const set = <K extends keyof ClientSettings>(key: K, value: ClientSettings[K]) =>
    setDraft((d) => ({ ...d, [key]: value }));

  const save = () => {
    logInfo(
      "settings saved",
      `url=${draft.url}`,
      `key=${draft.apiKey ? "set" : "EMPTY"}`,
      `history=${draft.sendHistory}`,
      `systemPrompt=${draft.systemPrompt ? `${draft.systemPrompt.length} chars` : "(none)"}`,
    );
    onSave({ ...draft, maxTokens: Math.max(1, Math.min(512, Math.round(draft.maxTokens) || 512)) });
    onClose();
  };

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-label="Settings">
        <h2>Settings</h2>

        <section>
          <h3>Connection</h3>
          <label className="field">
            <span>Coordinator URL</span>
            <input
              type="text"
              spellCheck={false}
              placeholder="http://127.0.0.1:8000"
              value={draft.url}
              onChange={(e) => set("url", e.target.value)}
            />
          </label>
          <label className="field">
            <span>API key</span>
            <input
              type="password"
              spellCheck={false}
              placeholder="coordinator DAIN_API_KEY"
              value={draft.apiKey}
              onChange={(e) => set("apiKey", e.target.value)}
            />
          </label>
        </section>

        <section>
          <h3>Generation</h3>
          <label className="field">
            <span>Max tokens per reply</span>
            <input
              type="number"
              min={1}
              max={512}
              value={draft.maxTokens}
              onChange={(e) => set("maxTokens", Number(e.target.value))}
            />
          </label>
          <label className="field toggle">
            <input
              type="checkbox"
              checked={draft.sendHistory}
              onChange={(e) => set("sendHistory", e.target.checked)}
            />
            <span className="toggle-label">
              Send conversation history
              <em>
                On: the model sees the whole transcript and can follow up. Off: each
                message is answered standalone (shorter prompts, no memory).
              </em>
            </span>
          </label>
        </section>

        <section>
          <h3>System prompt</h3>
          <label className="field">
            <span>
              Prepended to every request{" "}
              <button
                type="button"
                className="link"
                onClick={() => set("systemPrompt", DEFAULT_SYSTEM_PROMPT)}
              >
                reset to default
              </button>
            </span>
            <textarea
              rows={7}
              spellCheck={false}
              placeholder="(empty = no system prompt)"
              value={draft.systemPrompt}
              onChange={(e) => set("systemPrompt", e.target.value)}
            />
          </label>
        </section>

        <div className="modal-actions">
          <button type="button" className="secondary" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="primary" onClick={save}>
            Save settings
          </button>
        </div>
      </div>
    </div>
  );
}
