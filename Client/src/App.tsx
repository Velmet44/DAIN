import { useEffect, useRef, useState } from "react";
import { defaultApiKey, defaultApiUrl } from "./api";
import { useLocalStorage } from "./components";
import { ChatView } from "./chat";
import { DashboardView } from "./dashboard";
import { defaultModel, defaultMaxTokens } from "./api";
import { logInfo } from "./logs";

type Tab = "chat" | "nodes";

export default function App() {
  const [baseUrl, setBaseUrl] = useLocalStorage("dain:base_url", defaultApiUrl());
  const [apiKey, setApiKey] = useLocalStorage("dain:api_key", defaultApiKey());
  const [tab, setTab] = useState<Tab>("chat");

  useEffect(() => {
    logInfo(
      "client loaded",
      `env.url=${defaultApiUrl()}`,
      `env.key=${defaultApiKey() ? "set" : "EMPTY"}`,
      `env.model=${defaultModel() || "(none)"}`,
      `env.maxTokens=${defaultMaxTokens()}`,
    );
  }, []);

  const urlFirst = useRef(true);
  const keyFirst = useRef(true);
  useEffect(() => {
    if (urlFirst.current) {
      urlFirst.current = false;
      return;
    }
    logInfo(`coordinator url -> ${baseUrl}`);
  }, [baseUrl]);
  useEffect(() => {
    if (keyFirst.current) {
      keyFirst.current = false;
      return;
    }
    logInfo(`api key -> ${apiKey ? "set" : "cleared"}`);
  }, [apiKey]);

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