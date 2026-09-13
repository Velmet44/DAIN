/** localStorage persistence for conversations + client settings.
 *
 * Versioned keys so a future shape change can migrate instead of silently
 * dropping a user's chat history. Older unversioned keys (dain:base_url,
 * dain:api_key, dain:model) are migrated on first load, then left alone. */

import type { ClientSettings, Conversation } from "./types";
import { logInfo, logWarn } from "./logs";

const CHATS_KEY = "dain:chats:v1";
const SETTINGS_KEY = "dain:settings:v1";

const MAX_CHATS = 50;
const MAX_MESSAGES_PER_CHAT = 200;

export const DEFAULT_SYSTEM_PROMPT = [
  "You are a helpful, knowledgeable assistant running on DAIN, a distributed",
  "AI inference network. Answer accurately and directly.",
  "Prefer clear structure: short paragraphs, bullet lists for enumerations,",
  "fenced code blocks with a language tag for code.",
  "State uncertainty openly instead of guessing, and ask a clarifying question",
  "when the request is ambiguous. Keep answers as short as the question allows.",
].join(" ");

export function defaultSettings(url: string, apiKey: string, model: string): ClientSettings {
  return {
    url,
    apiKey,
    modelId: model,
    maxTokens: 512,
    systemPrompt: DEFAULT_SYSTEM_PROMPT,
    sendHistory: true,
  };
}

function readJson<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    logWarn(`${key}: unreadable, resetting`);
    return null;
  }
}

export function loadSettings(
  url: string,
  apiKey: string,
  model: string,
): ClientSettings {
  const stored = readJson<Partial<ClientSettings>>(SETTINGS_KEY);
  const settings = { ...defaultSettings(url, apiKey, model) };
  if (stored) {
    for (const key of Object.keys(settings) as (keyof ClientSettings)[]) {
      const value = stored[key];
      if (value !== undefined && value !== null && `${value}` !== "") {
        // The launcher bakes env defaults into the bundle: a fresh page load
        // must honor them over a stale stored URL/key from a previous run.
        if (key === "url" && url && value !== url) {
          logInfo("settings: env-provided URL wins over stored value");
          continue;
        }
        if (key === "apiKey" && apiKey && value !== apiKey) {
          logInfo("settings: env-provided API key wins over stored value");
          continue;
        }
        (settings[key] as unknown) = value;
      }
    }
    return settings;
  }
  // First run on the v1 schema: adopt user-edited values from the pre-sidebar
  // client (JSON-encoded [value, userEdited] pairs) so an upgrade never loses
  // the configured coordinator URL/key/model.
  for (const [key, legacy] of [
    ["url", "dain:base_url"],
    ["apiKey", "dain:api_key"],
    ["modelId", "dain:model"],
  ] as const) {
    try {
      const raw = localStorage.getItem(legacy);
      if (!raw) continue;
      const parsed = JSON.parse(raw) as [string, boolean];
      if (Array.isArray(parsed) && parsed[1] && typeof parsed[0] === "string" && parsed[0]) {
        logInfo(`settings: migrated ${legacy}`);
        (settings[key] as unknown) = parsed[0];
      }
    } catch {
      /* legacy value unreadable — ignore */
    }
  }
  return settings;
}

export function saveSettings(settings: ClientSettings): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    /* storage unavailable (private mode) */
  }
}

function sanitizeConversation(conv: Partial<Conversation>): Conversation | null {
  if (!conv || typeof conv.id !== "string" || !Array.isArray(conv.messages)) {
    return null;
  }
  const messages = conv.messages
    .filter((m) => m && typeof m.content === "string" && (m.role === "user" || m.role === "assistant"))
    .map((m, i) => ({
      id: typeof m.id === "string" ? m.id : `${conv.id}-${i}`,
      role: m.role,
      content: m.content,
      ts: typeof m.ts === "number" ? m.ts : Date.now(),
      ...(m.stats ? { stats: m.stats } : {}),
    }));
  return {
    id: conv.id,
    title: typeof conv.title === "string" && conv.title ? conv.title : "Untitled chat",
    createdAt: typeof conv.createdAt === "number" ? conv.createdAt : Date.now(),
    updatedAt: typeof conv.updatedAt === "number" ? conv.updatedAt : Date.now(),
    messages: messages.slice(-MAX_MESSAGES_PER_CHAT),
  };
}

export function loadConversations(): Conversation[] {
  const stored = readJson<Partial<Conversation>[]>(CHATS_KEY);
  if (!stored) return [];
  return stored
    .map(sanitizeConversation)
    .filter((c): c is Conversation => c !== null)
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, MAX_CHATS);
}

export function saveConversations(conversations: Conversation[]): void {
  try {
    localStorage.setItem(
      CHATS_KEY,
      JSON.stringify(conversations.slice(0, MAX_CHATS)),
    );
  } catch (e) {
    // Quota exceeded: drop the oldest half and retry once so a long session
    // never loses the *current* chat.
    if (conversations.length > 4) {
      logWarn("storage quota exceeded, pruning oldest chats");
      try {
        localStorage.setItem(
          CHATS_KEY,
          JSON.stringify(conversations.slice(0, Math.ceil(conversations.length / 2))),
        );
      } catch {
        /* give up quietly */
      }
    }
  }
}

export function newConversation(): Conversation {
  const now = Date.now();
  return {
    id: `c-${now.toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    title: "New chat",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

export function titleFromMessage(text: string): string {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return "New chat";
  return clean.length > 42 ? `${clean.slice(0, 42)}…` : clean;
}

export function relativeTime(ts: number): string {
  const secs = Math.floor((Date.now() - ts) / 1000);
  if (secs < 60) return "just now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}
