import { useEffect, useState } from "react";
import { clusterStatus, type ClusterStatus } from "./api";

interface Props {
  baseUrl: string;
  apiKey: string;
}

export function DashboardView({ baseUrl, apiKey }: Props) {
  const [status, setStatus] = useState<ClusterStatus | null>(null);
  const [error, setError] = useState("");
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => setTick((t) => t + 1), 3000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    let live = true;
    clusterStatus(baseUrl, apiKey)
      .then((s) => live && setStatus(s))
      .catch((err: Error) => live && setError(err.message));
    return () => {
      live = false;
    };
  }, [baseUrl, apiKey, tick]);

  return (
    <div className="view">
      <div className="toolbar">
        <span className="status">
          {error ? `error: ${error}` : status ? `active jobs: ${status.active_jobs}` : "loading…"}
        </span>
      </div>
      {status && (
        <>
          <table>
            <thead>
              <tr>
                <th>node</th>
                <th>state</th>
                <th>score</th>
                <th>gpu</th>
                <th>vram</th>
                <th>uptime</th>
                <th>fails</th>
                <th>conn</th>
              </tr>
            </thead>
            <tbody>
              {status.nodes.map((n) => (
                <tr key={n.node_id}>
                  <td>{n.node_id}</td>
                  <td>{n.state}</td>
                  <td>{typeof n.score === "number" ? n.score.toFixed(3) : "-"}</td>
                  <td>{n.gpu ?? "-"}</td>
                  <td>{n.vram_free_gb != null ? `${n.vram_free_gb.toFixed(1)} GB` : "-"}</td>
                  <td>
                    {n.uptime_ratio != null ? `${(n.uptime_ratio * 100).toFixed(0)}%` : "-"}
                  </td>
                  <td>
                    {n.failure_rate != null ? `${(n.failure_rate * 100).toFixed(0)}%` : "-"}
                  </td>
                  <td>{n.connected ? "✓" : "✗"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3>Placements</h3>
          <table>
            <thead>
              <tr>
                <th>time</th>
                <th>trigger</th>
                <th>model</th>
                <th>stages</th>
                <th>degraded</th>
                <th>nodes</th>
              </tr>
            </thead>
            <tbody>
              {status.placements.slice().reverse().map((p, i) => (
                <tr key={`${p.ts}-${i}`}>
                  <td>{new Date(p.ts * 1000).toLocaleTimeString()}</td>
                  <td>{p.trigger}</td>
                  <td>{p.model_id}</td>
                  <td>{p.k ?? "-"}</td>
                  <td>{p.degraded ? "yes" : "no"}</td>
                  <td>{p.node_ids.join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}