import { useState } from "react";
import { defaultApiKey, defaultApiUrl } from "./api";
import { useLocalStorage } from "./components";
import { ChatView } from "./chat";
import { DashboardView } from "./dashboard";

type Tab = "chat" | "nodes";

export default function App() {
  const [baseUrl, setBaseUrl] = useLocalStorage("dain:base_url", defaultApiUrl());
  const [apiKey, setApiKey] = useLocalStorage("dain:api_key", defaultApiKey());
  const [tab, setTab] = useState<Tab>("chat");

  return (
    <main>
      <nav>
        <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>
          chat
        </button>
        <button className={tab === "nodes" ? "active" : ""} onClick={() => setTab("nodes")}>
          nodes
        </button>
      </nav>
      {tab === "chat" ? (
        <ChatView baseUrl={baseUrl} apiKey={apiKey} setBaseUrl={setBaseUrl} setApiKey={setApiKey} />
      ) : (
        <DashboardView baseUrl={baseUrl} apiKey={apiKey} />
      )}
    </main>
  );
}