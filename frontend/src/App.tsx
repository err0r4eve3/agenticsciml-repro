import { FormEvent, useEffect, useMemo, useState } from "react";

type Benchmark = {
  name: string;
  path: string;
  paper_section: string;
  family: string;
  metric: string;
  fidelity_level: string;
  scientific_claim?: string;
};

type RunSummary = {
  run_id: string;
  status: string;
  run_dir: string;
  metadata: null | {
    run_state?: string;
    champion_node_id?: string;
    solution_count?: number;
    evidence_mode?: string;
    scientific_claim?: string;
  };
  trace_summary: null | {
    event_count?: number;
    quality_gate?: {
      passed: boolean;
      missing_event_types?: string[];
    };
  };
  leaderboard: Array<Record<string, string>>;
  artifacts: Array<{ path: string; kind: string; size_bytes?: number | null }>;
  error?: string;
};

type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  text: string;
};

type SolverResponse = {
  reply: string;
  actions: Array<{ type: string; payload?: unknown; run_id?: string }>;
  artifacts: Array<Record<string, unknown>>;
  warnings: string[];
  trace_refs: Array<Record<string, unknown>>;
};

type CodeServerPayload = {
  workspace: string;
  url: string;
  warnings: string[];
  command_hint: string;
};

const api = {
  async getBenchmarks(): Promise<Benchmark[]> {
    const payload = await getJson<{ benchmarks: Benchmark[] }>("/api/benchmarks");
    return payload.benchmarks;
  },
  async startMockRun(benchmark: string): Promise<RunSummary> {
    const id = `web-${new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 14)}`;
    return postJson<RunSummary>("/api/runs", {
      benchmark,
      mode: "mock",
      experiment_id: id,
      max_iterations: 0,
      parallel_mutations: 1,
      background: false
    });
  },
  async getRun(runId: string): Promise<RunSummary> {
    return getJson<RunSummary>(`/api/runs/${encodeURIComponent(runId)}`);
  },
  async askSolver(body: {
    message: string;
    active_run_id: string | null;
    selected_benchmark: string;
    mode: "mock" | "real" | "dry_run";
    workspace_scope: "repo" | "run" | "solution";
  }): Promise<SolverResponse> {
    return postJson<SolverResponse>("/api/solver/chat", body);
  },
  async getCodeServer(scope: "repo" | "run" | "solution", runId: string | null): Promise<CodeServerPayload> {
    const params = new URLSearchParams({ scope });
    if (runId) params.set("run_id", runId);
    return getJson<CodeServerPayload>(`/api/code-server/url?${params.toString()}`);
  }
};

export function App() {
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [selectedBenchmark, setSelectedBenchmark] = useState("function_approx");
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [activeRun, setActiveRun] = useState<RunSummary | null>(null);
  const [events, setEvents] = useState<string[]>([]);
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 1,
      role: "assistant",
      text: "选择 benchmark 后可启动 mock run、查看 trace summary，并打开 code-server 工作区。"
    }
  ]);
  const [mode, setMode] = useState<"mock" | "real" | "dry_run">("mock");
  const [workspaceScope, setWorkspaceScope] = useState<"repo" | "run" | "solution">("repo");
  const [codeServer, setCodeServer] = useState<CodeServerPayload | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getBenchmarks()
      .then((items) => {
        setBenchmarks(items);
        if (items.length && !items.some((item) => item.name === selectedBenchmark)) {
          setSelectedBenchmark(items[0].name);
        }
      })
      .catch((exc) => setError(String(exc)));
  }, []);

  useEffect(() => {
    if (!activeRunId) return;
    api.getRun(activeRunId).then(setActiveRun).catch((exc) => setError(String(exc)));
  }, [activeRunId]);

  useEffect(() => {
    if (!activeRunId) return;
    const source = new EventSource(`/api/runs/${encodeURIComponent(activeRunId)}/events?follow=false`);
    source.addEventListener("trace", (event) => {
      setEvents((current) => [event.data, ...current].slice(0, 40));
    });
    source.addEventListener("end", () => source.close());
    source.onerror = () => source.close();
    return () => source.close();
  }, [activeRunId]);

  useEffect(() => {
    api
      .getCodeServer(workspaceScope, activeRunId)
      .then(setCodeServer)
      .catch(() => setCodeServer(null));
  }, [workspaceScope, activeRunId]);

  const selected = useMemo(
    () => benchmarks.find((benchmark) => benchmark.name === selectedBenchmark),
    [benchmarks, selectedBenchmark]
  );

  async function startMockRun() {
    setBusy(true);
    setError(null);
    try {
      const run = await api.startMockRun(selectedBenchmark);
      setActiveRun(run);
      setActiveRunId(run.run_id);
      setMessages((current) => [
        ...current,
        { id: Date.now(), role: "assistant", text: `mock run 完成：${run.run_id}` }
      ]);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function sendMessage(event: FormEvent) {
    event.preventDefault();
    if (!message.trim()) return;
    const userMessage = message.trim();
    setMessage("");
    setMessages((current) => [...current, { id: Date.now(), role: "user", text: userMessage }]);
    try {
      const response = await api.askSolver({
        message: userMessage,
        active_run_id: activeRunId,
        selected_benchmark: selectedBenchmark,
        mode,
        workspace_scope: workspaceScope
      });
      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: "assistant",
          text: formatSolverResponse(response)
        }
      ]);
    } catch (exc) {
      setError(String(exc));
    }
  }

  return (
    <main className="workspace">
      <section className="chat-shell" aria-label="ChatUI 控制台">
        <div className="brand-row">
          <div>
            <h1>AgenticSciML</h1>
            <p>ChatUI 实验操作台</p>
          </div>
          <span className="status-dot">local</span>
        </div>
        <div className="field-stack">
          <label>
            Benchmark
            <select value={selectedBenchmark} onChange={(event) => setSelectedBenchmark(event.target.value)}>
              {benchmarks.map((benchmark) => (
                <option key={benchmark.name} value={benchmark.name}>
                  {benchmark.name}
                </option>
              ))}
            </select>
          </label>
          <div className="segmented" aria-label="运行模式">
            {(["mock", "dry_run", "real"] as const).map((item) => (
              <button
                key={item}
                className={mode === item ? "selected" : ""}
                type="button"
                onClick={() => setMode(item)}
              >
                {item}
              </button>
            ))}
          </div>
          <button className="primary-action" type="button" disabled={busy} onClick={startMockRun}>
            {busy ? "running..." : "Run mock root"}
          </button>
        </div>
        <div className="messages">
          {messages.map((item) => (
            <article className={`message ${item.role}`} key={item.id}>
              <span>{item.role}</span>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
        <form className="composer" onSubmit={sendMessage}>
          <input
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            placeholder="例如：解释 trace / 打开 champion / 跑 mock"
          />
          <button type="submit">Send</button>
        </form>
      </section>

      <section className="dashboard" aria-label="实验状态">
        <header className="dashboard-header">
          <div>
            <p className="eyebrow">Run dashboard</p>
            <h2>{activeRunId ?? "No active run"}</h2>
          </div>
          <button type="button" disabled={!activeRunId} onClick={() => activeRunId && api.getRun(activeRunId).then(setActiveRun)}>
            Refresh
          </button>
        </header>
        {error ? <div className="error-line">{error}</div> : null}
        <div className="metric-grid">
          <Metric label="status" value={activeRun?.status ?? "idle"} />
          <Metric label="benchmark" value={selected?.name ?? selectedBenchmark} />
          <Metric label="fidelity" value={selected?.fidelity_level ?? "unknown"} />
          <Metric
            label="quality gate"
            value={String(activeRun?.trace_summary?.quality_gate?.passed ?? "pending")}
          />
        </div>
        <section className="data-region">
          <h3>Leaderboard</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>rank</th>
                  <th>node</th>
                  <th>metric</th>
                  <th>score</th>
                  <th>status</th>
                </tr>
              </thead>
              <tbody>
                {(activeRun?.leaderboard ?? []).slice(0, 6).map((row) => (
                  <tr key={`${row.rank}-${row.node_id}`}>
                    <td>{row.rank}</td>
                    <td>{row.node_id}</td>
                    <td>{row.metric}</td>
                    <td>{row.score}</td>
                    <td>{row.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
        <section className="data-region">
          <h3>Artifacts</h3>
          <div className="artifact-list">
            {(activeRun?.artifacts ?? []).slice(0, 12).map((artifact) => (
              <span key={artifact.path}>{artifact.path}</span>
            ))}
          </div>
        </section>
        <section className="data-region">
          <h3>Trace events</h3>
          <div className="event-log">
            {events.length === 0 ? <p>No events loaded.</p> : null}
            {events.map((event, index) => (
              <code key={`${index}-${event.slice(0, 12)}`}>{event}</code>
            ))}
          </div>
        </section>
      </section>

      <section className="code-panel" aria-label="code-server sidecar">
        <header>
          <div>
            <p className="eyebrow">VS Code Web</p>
            <h2>code-server</h2>
          </div>
          <select value={workspaceScope} onChange={(event) => setWorkspaceScope(event.target.value as typeof workspaceScope)}>
            <option value="repo">repo</option>
            <option value="run">run</option>
            <option value="solution">solution</option>
          </select>
        </header>
        {codeServer ? (
          <>
            <div className="workspace-path">{codeServer.workspace}</div>
            <a className="code-link" href={codeServer.url} target="_blank" rel="noreferrer">
              Open VS Code Web
            </a>
            <pre>{codeServer.command_hint}</pre>
            <ul>
              {codeServer.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </>
        ) : (
          <p className="muted">选择 active run 后可打开 run 或 solution workspace。</p>
        )}
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json() as Promise<T>;
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json() as Promise<T>;
}

function formatSolverResponse(response: SolverResponse) {
  const lines = [response.reply];
  if (response.actions.length) lines.push(`actions: ${response.actions.map((action) => action.type).join(", ")}`);
  if (response.warnings.length) lines.push(`warnings: ${response.warnings.join(" / ")}`);
  if (response.artifacts.length) lines.push(`artifacts: ${response.artifacts.length}`);
  return lines.join("\n");
}
