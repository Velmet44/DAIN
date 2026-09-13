/** Client-side domain types: conversations, messages, per-response stats. */

export type Role = "user" | "assistant";

export interface MsgStats {
  model: string;
  /** Wall-clock from request send to the first streamed token (ms). */
  ttftMs: number | null;
  /** Wall-clock from request send to the final frame (ms). */
  totalMs: number;
  /** Generated tokens (server usage when present, else counted token frames). */
  tokens: number;
  stopped?: boolean;
  error?: string;
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  streaming?: boolean;
  stats?: MsgStats;
  ts: number;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
}

export interface ClientSettings {
  url: string;
  apiKey: string;
  modelId: string;
  maxTokens: number;
  /** Prepend a system prompt to every request. */
  systemPrompt: string;
  /** Send the full conversation transcript (not just the latest message). */
  sendHistory: boolean;
}
