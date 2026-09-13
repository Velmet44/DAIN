import { useState } from "react";
import type { Conversation } from "./types";
import { relativeTime } from "./storage";
import { logInfo } from "./logs";

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onOpenSettings: () => void;
  onOpenNodes: () => void;
  nodesView: boolean;
  connected: boolean;
}

/** Left rail: new chat, saved conversation list with two-step delete,
 * settings + cluster dashboard entries. */
export function Sidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  onOpenSettings,
  onOpenNodes,
  nodesView,
  connected,
}: Props) {
  const [confirmId, setConfirmId] = useState<string | null>(null);

  return (
    <aside className="sidebar">
      <div className="side-head">
        <span className="brand">
          DAIN <span className="brand-sub">client</span>
        </span>
        <span className={`dot ${connected ? "on" : "off"}`} title={connected ? "coordinator reachable" : "coordinator unreachable"} />
      </div>

      <button className="new-chat" type="button" onClick={onNew}>
        + New chat
      </button>

      <nav className="chat-list">
        {conversations.length === 0 && (
          <p className="side-hint">No conversations yet — say hi.</p>
        )}
        {conversations.map((conv) => {
          const active = conv.id === activeId && !nodesView;
          return (
            <div key={conv.id} className={`chat-item ${active ? "active" : ""}`}>
              <button type="button" className="chat-open" onClick={() => onSelect(conv.id)}>
                <span className="chat-title">{conv.title}</span>
                <span className="chat-when">{relativeTime(conv.updatedAt)}</span>
              </button>
              {confirmId === conv.id ? (
                <button
                  type="button"
                  className="chat-del sure"
                  title="Click again to delete"
                  onClick={() => {
                    logInfo(`delete chat ${conv.id}`);
                    setConfirmId(null);
                    onDelete(conv.id);
                  }}
                >
                  delete?
                </button>
              ) : (
                <button
                  type="button"
                  className="chat-del"
                  title="Delete chat"
                  onClick={() => {
                    setConfirmId(conv.id);
                    window.setTimeout(() => {
                      setConfirmId((cur) => (cur === conv.id ? null : cur));
                    }, 2500);
                  }}
                >
                  ✕
                </button>
              )}
            </div>
          );
        })}
      </nav>

      <div className="side-foot">
        <button type="button" className={nodesView ? "foot-btn active" : "foot-btn"} onClick={onOpenNodes}>
          ⬡ Cluster nodes
        </button>
        <button type="button" className="foot-btn" onClick={onOpenSettings}>
          ⚙ Settings
        </button>
      </div>
    </aside>
  );
}
