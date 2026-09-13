import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { defaultApiKey, defaultApiUrl, defaultModel, fetchMeta } from "./api";
import { applyMeta } from "./storage";
import { ChatView } from "./chat";
import { DashboardView } from "./dashboard";
import { Sidebar } from "./sidebar";
import { SettingsModal } from "./settingsModal";
import {
  loadConversations,
  loadSettings,
  newConversation,
  saveConversations,
  saveSettings,
} from "./storage";
import type { ChatMessage, ClientSettings, Conversation } from "./types";
import { logInfo } from "./logs";

export default function App() {
  const [settings, setSettings] = useState<ClientSettings>(() =>
    loadSettings(defaultApiUrl(), defaultApiKey(), defaultModel()),
  );
  const [conversations, setConversations] = useState<Conversation[]>(() =>
    loadConversations(),
  );
  const [activeId, setActiveId] = useState<string | null>(
    () => loadConversations()[0]?.id ?? null,
  );
  const [nodesView, setNodesView] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [connected, setConnected] = useState(false);
  const saveTimer = useRef<number | undefined>(undefined);

  const active = useMemo(
    () => conversations.find((c) => c.id === activeId) ?? null,
    [conversations, activeId],
  );

  const patchConversations = useCallback((patcher: (list: Conversation[]) => Conversation[]) => {
    setConversations((list) => {
      const next = patcher(list);
      // Coalesce rapid streaming patches into one storage write per second.
      window.clearTimeout(saveTimer.current);
      saveTimer.current = window.setTimeout(() => saveConversations(next), 400);
      return next;
    });
  }, []);

  const patchActive = useCallback(
    (patch: {
      title?: string;
      appendMessage?: ChatMessage;
      replaceMessage?: ChatMessage;
    }) => {
      patchConversations((list) =>
        list.map((conv) => {
          if (conv.id !== activeId) return conv;
          let messages = conv.messages;
          if (patch.appendMessage) messages = [...messages, patch.appendMessage];
          if (patch.replaceMessage) {
            const target = patch.replaceMessage.id;
            const idx = messages.findIndex((m) => m.id === target);
            messages =
              idx >= 0
                ? messages.map((m, i) => (i === idx ? patch.replaceMessage! : m))
                : [...messages, patch.replaceMessage];
          }
          return {
            ...conv,
            title: patch.title ?? conv.title,
            messages,
            updatedAt: Date.now(),
          };
        }),
      );
    },
    [activeId, patchConversations],
  );

  const startNewChat = useCallback(() => {
    setNodesView(false);
    const conv = newConversation();
    patchConversations((list) => [conv, ...list]);
    setActiveId(conv.id);
    logInfo(`new chat ${conv.id}`);
  }, [patchConversations]);

  const deleteChat = useCallback(
    (id: string) => {
      patchConversations((list) => list.filter((c) => c.id !== id));
      setActiveId((cur) => {
        if (cur !== id) return cur;
        const remaining = conversations.filter((c) => c.id !== id);
        return remaining[0]?.id ?? null;
      });
    },
    [conversations, patchConversations],
  );

  const saveSettingsNow = useCallback((next: ClientSettings) => {
    setSettings(next);
    saveSettings(next);
  }, []);

  // S23: adopt published coordinator metadata (URL/key/model) on every open,
  // respecting user-edited values (see applyMeta).
  useEffect(() => {
    fetchMeta().then((meta) => {
      if (!meta) return;
      setSettings((current) => applyMeta(current, meta));
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    logInfo(
      "client loaded",
      `url=${settings.url}`,
      `key=${settings.apiKey ? "set" : "EMPTY"}`,
      `model=${settings.modelId || "(none)"}`,
      `history=${settings.sendHistory}`,
      `systemPrompt=${settings.systemPrompt ? `${settings.systemPrompt.length} chars` : "(none)"}`,
      `chats=${conversations.length}`,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Ensure at least one conversation exists once the user starts chatting.
  const showChat = !nodesView;
  const effectiveActive = active;

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        nodesView={nodesView}
        connected={connected}
        onSelect={(id) => {
          setActiveId(id);
          setNodesView(false);
        }}
        onNew={startNewChat}
        onDelete={deleteChat}
        onOpenSettings={() => setSettingsOpen(true)}
        onOpenNodes={() => setNodesView(true)}
      />
      <main className="main">
        {showChat ? (
          effectiveActive ? (
            <ChatView
              conv={effectiveActive}
              settings={settings}
              onModelPicked={(modelId) => saveSettingsNow({ ...settings, modelId })}
              onPatch={patchActive}
              onConnected={setConnected}
            />
          ) : (
            <div className="chat-view">
              <div className="welcome">
                <h1>DAIN</h1>
                <p>Start a new chat from the sidebar to begin.</p>
                <button type="button" className="primary" onClick={startNewChat}>
                  + New chat
                </button>
              </div>
            </div>
          )
        ) : (
          <DashboardView baseUrl={settings.url} apiKey={settings.apiKey} />
        )}
      </main>
      {settingsOpen && (
        <SettingsModal
          settings={settings}
          onSave={saveSettingsNow}
          onClose={() => setSettingsOpen(false)}
        />
      )}
    </div>
  );
}
