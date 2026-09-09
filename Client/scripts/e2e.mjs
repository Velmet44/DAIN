// E2E: boot a real local Sim cluster (coordinator + nodes, tiny model), stream a
// completion through the public /v1 API the way the browser does, and fail if no
// token frames + final frame + [DONE] arrive. This is the Client-stage gate.
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { execSync } from "node:child_process";

const REPO_ROOT = fileURLToPath(new URL("../../", import.meta.url));
const API_KEY = "dain-dev-key";
const MODEL = "dain-tiny-16L";

async function getJson(url, key = API_KEY) {
  const resp = await fetch(url, { headers: { "X-API-Key": key } });
  if (!resp.ok) throw new Error(`${url} -> ${resp.status}`);
  return resp.json();
}

async function streamCompletions(base) {
  const resp = await fetch(`${base}/v1/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream", "X-API-Key": API_KEY },
    body: JSON.stringify({ model_id: MODEL, prompt: "hello from e2e", max_tokens: 16, stream: true }),
  });
  if (!resp.ok) throw new Error(`completions -> ${resp.status}: ${await resp.text()}`);
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let tokens = 0;
  let sawFinal = false;
  let sawDone = false;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6).trim();
      if (payload === "[DONE]") {
        sawDone = true;
        continue;
      }
      const frame = JSON.parse(payload);
      if (frame.type === "token") tokens++;
      if (frame.type === "final") sawFinal = true;
    }
  }
  return { tokens, sawFinal, sawDone };
}

function killTree(pid) {
  try {
    execSync(`taskkill /PID ${pid} /T /F`, { stdio: "ignore" });
  } catch {
    /* already gone */
  }
}

async function waitForOnline(base) {
  const deadline = Date.now() + 60_000;
  for (;;) {
    try {
      const health = await getJson(`${base}/v1/nodes`);
      if (health.nodes?.some((n) => n.state === "online" && n.connected)) return health;
    } catch {
      /* coordinator still booting */
    }
    if (Date.now() > deadline) throw new Error("no ONLINE node within 60s");
    await new Promise((r) => setTimeout(r, 750));
  }
}

async function run() {
  const child = spawn(
    "uv",
    ["run", "--project", "Sim", "python", "-m", "dain_sim.cluster", "--nodes", "2", "--chat", "--max-tokens", "16"],
    { cwd: REPO_ROOT, stdio: ["pipe", "pipe", "inherit"], windowsHide: true, env: { ...process.env, PYTHONUNBUFFERED: "1" } },
  );

  const base = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("coordinator port line never flushed")), 120_000);
    let stdout = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
      const match = /coordinator on :(\d+)/.exec(stdout);
      if (match) {
        clearTimeout(timer);
        resolve(`http://127.0.0.1:${match[1]}`);
      }
    });
    child.on("exit", () => {
      clearTimeout(timer);
      reject(new Error(`cluster exited early: ${stdout.slice(-300)}`));
    });
  });

  try {
    const health = await waitForOnline(base);
    const models = await getJson(`${base}/v1/models`);
    if (!models.models?.length) throw new Error("no models listed");

    let outcome;
    const deadline = Date.now() + 45_000;
    for (;;) {
      try {
        outcome = await streamCompletions(base);
        break;
      } catch (err) {
        if (err.message.startsWith("completions -> 429") && Date.now() < deadline) {
          await new Promise((r) => setTimeout(r, 1000));
          continue;
        }
        throw err;
      }
    }
    if (outcome.tokens < 1) throw new Error("no token frames streamed");
    if (!outcome.sawFinal) throw new Error("no final frame");
    if (!outcome.sawDone) throw new Error("stream did not end with [DONE]");
    console.log(`e2e OK: ${outcome.tokens} tokens, final=${outcome.sawFinal}, DONE=${outcome.sawDone}, nodes=${health.nodes.length}`);
  } finally {
    child.stdin.end();
    await new Promise((r) => setTimeout(r, 1500));
    killTree(child.pid);
  }
}

run()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(`e2e FAIL: ${err.message}`);
    process.exit(1);
  });