import { FormEvent, ReactNode, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Bot,
  ChevronRight,
  Code2,
  Database,
  FileText,
  FolderTree,
  PanelRightClose,
  PanelRightOpen,
  Play,
  RefreshCw,
  Send,
  ShieldCheck,
  Sparkles,
  TerminalSquare
} from "lucide-react";

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
  artifacts: ArtifactEntry[];
  error?: string;
};

type ArtifactEntry = {
  path: string;
  kind: string;
  size_bytes?: number | null;
};

type ArtifactPayload =
  | {
      path: string;
      kind: "directory";
      entries: ArtifactEntry[];
    }
  | {
      path: string;
      kind: "file";
      size_bytes: number;
      content: string;
    };

type RunMode = "mock" | "real" | "dry_run";
type WorkspaceScope = "repo" | "run" | "solution";

type AgentMessage = {
  id: number;
  role: "user" | "assistant";
  text: string;
  response?: SolverResponse;
};

type SolverAction = {
  type: string;
  payload?: {
    benchmark?: string;
    mode?: RunMode;
    background?: boolean;
    scope?: WorkspaceScope;
    [key: string]: unknown;
  };
  run_id?: string;
};

type SolverResponse = {
  reply: string;
  actions: SolverAction[];
  artifacts: Array<Record<string, unknown>>;
  warnings: string[];
  trace_refs: Array<Record<string, unknown>>;
};

type CodeServerPayload = {
  scope: WorkspaceScope;
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
  async getRuns(): Promise<RunSummary[]> {
    const payload = await getJson<{ runs: RunSummary[] }>("/api/runs");
    return payload.runs;
  },
  async startRun(body: {
    benchmark: string;
    mode: RunMode;
    experiment_id?: string;
    max_iterations?: number;
    parallel_mutations?: number;
    background?: boolean;
  }): Promise<RunSummary> {
    const id = body.experiment_id ?? `web-${new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 14)}`;
    await postJson<unknown>("/api/runs", {
      benchmark: body.benchmark,
      mode: body.mode,
      experiment_id: id,
      max_iterations: body.max_iterations ?? 0,
      parallel_mutations: body.parallel_mutations ?? 1,
      background: body.background ?? false
    });
    return api.getRun(id);
  },
  async resumeRun(runId: string, benchmark: string, mode: RunMode): Promise<RunSummary> {
    await postJson<unknown>(`/api/runs/${encodeURIComponent(runId)}/resume`, {
      benchmark,
      mode,
      experiment_id: runId,
      max_iterations: 1,
      parallel_mutations: 1,
      background: true
    });
    return api.getRun(runId);
  },
  async getRun(runId: string): Promise<RunSummary> {
    return getJson<RunSummary>(`/api/runs/${encodeURIComponent(runId)}`);
  },
  async getArtifact(runId: string, artifactPath: string): Promise<ArtifactPayload> {
    return getJson<ArtifactPayload>(
      `/api/runs/${encodeURIComponent(runId)}/artifacts/${artifactPath
        .split("/")
        .map(encodeURIComponent)
        .join("/")}`
    );
  },
  async askSolver(body: {
    message: string;
    active_run_id: string | null;
    selected_benchmark: string;
    mode: RunMode;
    workspace_scope: WorkspaceScope;
  }): Promise<SolverResponse> {
    return postJson<SolverResponse>("/api/solver/chat", body);
  },
  async getCodeServer(scope: WorkspaceScope, runId: string | null): Promise<CodeServerPayload> {
    const params = new URLSearchParams({ scope });
    if (runId) params.set("run_id", runId);
    return getJson<CodeServerPayload>(`/api/code-server/url?${params.toString()}`);
  }
};

export function App() {
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedBenchmark, setSelectedBenchmark] = useState("function_approx");
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [activeRun, setActiveRun] = useState<RunSummary | null>(null);
  const [events, setEvents] = useState<string[]>([]);
  const [traceFilter, setTraceFilter] = useState("");
  const [selectedArtifactPath, setSelectedArtifactPath] = useState("");
  const [artifactPayload, setArtifactPayload] = useState<ArtifactPayload | null>(null);
  const [message, setMessage] = useState("");
  const [mainPrompt, setMainPrompt] = useState("");
  const [messages, setMessages] = useState<AgentMessage[]>([
    {
      id: 1,
      role: "assistant",
      text: "选择 benchmark 后可启动 mock run、解释 trace、浏览 artifact，并打开 code-server 工作区。"
    }
  ]);
  const [mode, setMode] = useState<RunMode>("mock");
  const [workspaceScope, setWorkspaceScope] = useState<WorkspaceScope>("repo");
  const [codeServer, setCodeServer] = useState<CodeServerPayload | null>(null);
  const [agentCollapsed, setAgentCollapsed] = useState(false);
  const [pendingRealAction, setPendingRealAction] = useState<SolverAction | null>(null);
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
    refreshRuns().catch((exc) => setError(String(exc)));
  }, []);

  useEffect(() => {
    if (!activeRunId) return;
    refreshActiveRun(activeRunId).catch((exc) => setError(String(exc)));
  }, [activeRunId]);

  useEffect(() => {
    if (!activeRunId) return;
    setEvents([]);
    const source = new EventSource(`/api/runs/${encodeURIComponent(activeRunId)}/events?follow=false`);
    source.addEventListener("trace", (event) => {
      setEvents((current) => [event.data, ...current].slice(0, 80));
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

  useEffect(() => {
    if (!activeRunId || !selectedArtifactPath) {
      setArtifactPayload(null);
      return;
    }
    api
      .getArtifact(activeRunId, selectedArtifactPath)
      .then(setArtifactPayload)
      .catch((exc) => setError(String(exc)));
  }, [activeRunId, selectedArtifactPath]);

  const selected = useMemo(
    () => benchmarks.find((benchmark) => benchmark.name === selectedBenchmark),
    [benchmarks, selectedBenchmark]
  );
  const filteredEvents = useMemo(() => {
    const needle = traceFilter.trim().toLowerCase();
    if (!needle) return events;
    return events.filter((event) => event.toLowerCase().includes(needle));
  }, [events, traceFilter]);
  const runState = activeRun?.metadata?.run_state ?? activeRun?.status ?? "idle";
  const qualityGate = activeRun?.trace_summary?.quality_gate?.passed;

  async function refreshRuns() {
    const payload = await api.getRuns();
    setRuns(payload);
  }

  async function refreshActiveRun(runId = activeRunId) {
    if (!runId) return;
    const run = await api.getRun(runId);
    setActiveRun(run);
    setRuns((current) => [run, ...current.filter((item) => item.run_id !== run.run_id)]);
  }

  async function refreshAll() {
    setError(null);
    await refreshRuns();
    await refreshActiveRun();
  }

  async function startRun(nextMode: RunMode = mode, background = false) {
    if (nextMode === "real") {
      setPendingRealAction({
        type: "start_run",
        payload: { benchmark: selectedBenchmark, mode: "real", background }
      });
      setAgentCollapsed(false);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const run = await api.startRun({
        benchmark: selectedBenchmark,
        mode: nextMode,
        max_iterations: 0,
        parallel_mutations: 1,
        background
      });
      setActiveRun(run);
      setActiveRunId(run.run_id);
      await refreshRuns();
      if (background) {
        window.setTimeout(() => {
          refreshActiveRun(run.run_id).catch((exc) => setError(String(exc)));
        }, 1200);
      }
      addAssistantMessage(`${nextMode} run 已登记：${run.run_id}`);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function submitAgentMessage(rawMessage: string, overrides: { mode?: RunMode; workspaceScope?: WorkspaceScope } = {}) {
    if (!rawMessage.trim()) return;
    const userMessage = rawMessage.trim();
    setMessages((current) => [...current, { id: Date.now(), role: "user", text: userMessage }]);
    setBusy(true);
    setError(null);
    try {
      const requestWorkspaceScope = overrides.workspaceScope ?? workspaceScope;
      const response = await api.askSolver({
        message: userMessage,
        active_run_id: activeRunId,
        selected_benchmark: selectedBenchmark,
        mode: overrides.mode ?? mode,
        workspace_scope: requestWorkspaceScope
      });
      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: "assistant",
          text: response.reply,
          response
        }
      ]);
      if (requestWorkspaceScope !== workspaceScope) setWorkspaceScope(requestWorkspaceScope);
      await dispatchSolverActions(response.actions);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function sendMessage(event: FormEvent) {
    event.preventDefault();
    if (!message.trim()) return;
    const userMessage = message;
    setMessage("");
    await submitAgentMessage(userMessage);
  }

  async function sendMainPrompt(event: FormEvent) {
    event.preventDefault();
    if (!mainPrompt.trim()) return;
    const userMessage = mainPrompt;
    setMainPrompt("");
    await submitAgentMessage(userMessage);
  }

  async function dispatchSolverActions(actions: SolverAction[]) {
    for (const action of actions) {
      if (action.type === "start_run") {
        const actionMode = action.payload?.mode ?? mode;
        if (actionMode === "real") {
          setPendingRealAction(action);
          continue;
        }
        await startRun(actionMode, Boolean(action.payload?.background));
      }
      if (action.type === "resume_run" && action.run_id) {
        if (mode === "real") {
          setPendingRealAction(action);
          continue;
        }
        const run = await api.resumeRun(action.run_id, selectedBenchmark, mode);
        setActiveRun(run);
        setActiveRunId(run.run_id);
        await refreshRuns();
      }
      if (action.type === "open_code_server") {
        const nextScope = action.payload?.scope;
        if (nextScope === "repo" || nextScope === "run" || nextScope === "solution") {
          setWorkspaceScope(nextScope);
        }
      }
      if (action.type === "summarize_artifact") {
        const artifactPath = action.payload?.path;
        if (typeof artifactPath === "string") {
          setSelectedArtifactPath(artifactPath);
        }
      }
    }
  }

  function addAssistantMessage(text: string) {
    setMessages((current) => [...current, { id: Date.now(), role: "assistant", text }]);
  }

  function selectRun(run: RunSummary) {
    setActiveRun(run);
    setActiveRunId(run.run_id);
    setSelectedArtifactPath("");
  }

  return (
    <main className={`workbench ${agentCollapsed ? "agent-is-collapsed" : ""}`}>
      <section className="chatui-column" aria-label="完整 ChatUI 页面">
        <TopBar
          activeRunId={activeRunId}
          busy={busy}
          mode={mode}
          qualityGate={qualityGate}
          runState={runState}
          selectedBenchmark={selectedBenchmark}
          benchmarks={benchmarks}
          onModeChange={setMode}
          onRefresh={refreshAll}
          onRun={() => startRun(mode, false)}
          onSelectBenchmark={setSelectedBenchmark}
        />
        {error ? <div className="error-line">{error}</div> : null}
        <ChatUIPage
          activeRun={activeRun}
          activeRunId={activeRunId}
          artifactPayload={artifactPayload}
          busy={busy}
          events={filteredEvents}
          filter={traceFilter}
          mainPrompt={mainPrompt}
          messages={messages}
          runs={runs}
          selected={selected}
          selectedArtifactPath={selectedArtifactPath}
          onFilterChange={setTraceFilter}
          onOpenArtifacts={() => setSelectedArtifactPath("trace_summary.json")}
          onOpenCode={() => setWorkspaceScope("solution")}
          onPromptChange={setMainPrompt}
          onQuickPrompt={(text, options) => submitAgentMessage(text, options)}
          onRefreshRuns={refreshRuns}
          onSelectArtifact={setSelectedArtifactPath}
          onSelectRun={selectRun}
          onStartMock={() => startRun("mock", false)}
          onSubmitPrompt={sendMainPrompt}
        />
      </section>
      <section className="vscode-column" aria-label="VS Code Web 与 ChatUI 侧边栏">
        <CodeView
          activeRunId={activeRunId}
          codeServer={codeServer}
          scope={workspaceScope}
          onScopeChange={setWorkspaceScope}
        />
        <AgentPanel
          collapsed={agentCollapsed}
          message={message}
          messages={messages}
          mode={mode}
          pendingRealAction={pendingRealAction}
          busy={busy}
          onChangeMessage={setMessage}
          onSend={sendMessage}
          onToggle={() => setAgentCollapsed((current) => !current)}
          onDismissRealAction={() => setPendingRealAction(null)}
        />
      </section>
    </main>
  );
}

function TopBar({
  activeRunId,
  benchmarks,
  busy,
  mode,
  qualityGate,
  runState,
  selectedBenchmark,
  onModeChange,
  onRefresh,
  onRun,
  onSelectBenchmark
}: {
  activeRunId: string | null;
  benchmarks: Benchmark[];
  busy: boolean;
  mode: RunMode;
  qualityGate: boolean | undefined;
  runState: string;
  selectedBenchmark: string;
  onModeChange: (mode: RunMode) => void;
  onRefresh: () => void;
  onRun: () => void;
  onSelectBenchmark: (benchmark: string) => void;
}) {
  return (
    <header className="topbar">
      <div className="topbar-title">
        <h1>AgenticSciML</h1>
        <span>本地实验工作台</span>
      </div>
      <div className="topbar-controls">
        <label className="compact-field">
          <span>Benchmark</span>
          <select value={selectedBenchmark} onChange={(event) => onSelectBenchmark(event.target.value)}>
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
              onClick={() => onModeChange(item)}
            >
              {item}
            </button>
          ))}
        </div>
        <StatusBadge tone={qualityGate ? "good" : qualityGate === false ? "bad" : "neutral"}>
          gate {qualityGate === undefined ? "pending" : qualityGate ? "pass" : "fail"}
        </StatusBadge>
        <StatusBadge tone={runState === "idle" ? "neutral" : "info"}>{runState}</StatusBadge>
        <button className="icon-text-button" disabled={busy} type="button" onClick={onRun}>
          <Play size={15} />
          Run
        </button>
        <button className="icon-button" type="button" onClick={onRefresh} title="刷新">
          <RefreshCw size={16} />
        </button>
      </div>
      <div className="active-run">
        <span>active run</span>
        <strong>{activeRunId ?? "none"}</strong>
      </div>
    </header>
  );
}

function ChatUIPage({
  activeRun,
  activeRunId,
  artifactPayload,
  busy,
  events,
  filter,
  mainPrompt,
  messages,
  runs,
  selected,
  selectedArtifactPath,
  onFilterChange,
  onOpenArtifacts,
  onOpenCode,
  onPromptChange,
  onQuickPrompt,
  onRefreshRuns,
  onSelectArtifact,
  onSelectRun,
  onStartMock,
  onSubmitPrompt
}: {
  activeRun: RunSummary | null;
  activeRunId: string | null;
  artifactPayload: ArtifactPayload | null;
  busy: boolean;
  events: string[];
  filter: string;
  mainPrompt: string;
  messages: AgentMessage[];
  runs: RunSummary[];
  selected?: Benchmark;
  selectedArtifactPath: string;
  onFilterChange: (value: string) => void;
  onOpenArtifacts: () => void;
  onOpenCode: () => void;
  onPromptChange: (value: string) => void;
  onQuickPrompt: (text: string, options?: { mode?: RunMode; workspaceScope?: WorkspaceScope }) => void;
  onRefreshRuns: () => void;
  onSelectArtifact: (value: string) => void;
  onSelectRun: (run: RunSummary) => void;
  onStartMock: () => void;
  onSubmitPrompt: (event: FormEvent) => void;
}) {
  return (
    <section className="chat-page">
      <CommandCenter
        activeRunId={activeRun?.run_id ?? null}
        busy={busy}
        mainPrompt={mainPrompt}
        onChange={onPromptChange}
        onQuickPrompt={onQuickPrompt}
        onSubmit={onSubmitPrompt}
        selectedBenchmark={selected?.name ?? "function_approx"}
      />
      <DataRegion title="ChatUI conversation">
        <ChatTranscript messages={messages} />
      </DataRegion>
      <DashboardView
        activeRun={activeRun}
        events={events}
        onOpenArtifacts={onOpenArtifacts}
        onOpenCode={onOpenCode}
        selected={selected}
      />
      <RunsView
        activeRunId={activeRunId}
        runs={runs}
        onRefresh={onRefreshRuns}
        onSelectRun={onSelectRun}
        onStartMock={onStartMock}
      />
      <ArtifactsView
        activeRun={activeRun}
        artifactPayload={artifactPayload}
        events={events}
        filter={filter}
        selectedArtifactPath={selectedArtifactPath}
        onFilterChange={onFilterChange}
        onSelectArtifact={onSelectArtifact}
      />
    </section>
  );
}

function DashboardView({
  activeRun,
  events,
  onOpenArtifacts,
  onOpenCode,
  selected
}: {
  activeRun: RunSummary | null;
  events: string[];
  onOpenArtifacts: () => void;
  onOpenCode: () => void;
  selected?: Benchmark;
}) {
  const metadata = activeRun?.metadata;
  const qualityGate = activeRun?.trace_summary?.quality_gate?.passed;
  return (
    <div className="view-stack">
      <section className="section-head">
        <div>
          <p className="eyebrow">Run dashboard</p>
          <h2>{activeRun?.run_id ?? "尚未选择 run"}</h2>
        </div>
        <div className="section-actions">
          <button type="button" onClick={onOpenArtifacts}>
            <FolderTree size={15} />
            Artifacts
          </button>
          <button type="button" onClick={onOpenCode}>
            <Code2 size={15} />
            Code
          </button>
        </div>
      </section>
      <div className="metric-grid">
        <Metric label="status" value={activeRun?.status ?? "idle"} />
        <Metric label="run_state" value={metadata?.run_state ?? "pending"} />
        <Metric label="benchmark" value={selected?.name ?? "unknown"} />
        <Metric label="fidelity" value={selected?.fidelity_level ?? "unknown"} />
        <Metric label="scientific_claim" value={metadata?.scientific_claim ?? selected?.scientific_claim ?? "unknown"} />
        <Metric label="quality_gate" value={String(qualityGate ?? "pending")} />
        <Metric label="champion" value={metadata?.champion_node_id ?? "none"} />
        <Metric label="solutions" value={String(metadata?.solution_count ?? 0)} />
      </div>
      <div className="split-grid">
        <DataRegion title="Leaderboard">
          <Leaderboard rows={activeRun?.leaderboard ?? []} />
        </DataRegion>
        <DataRegion title="Trace preview">
          <TraceList events={events.slice(0, 8)} emptyText="尚无 trace event。" />
        </DataRegion>
      </div>
      <DataRegion title="Artifacts">
        <ArtifactChips artifacts={(activeRun?.artifacts ?? []).slice(0, 18)} />
      </DataRegion>
    </div>
  );
}

function CommandCenter({
  activeRunId,
  busy,
  mainPrompt,
  onChange,
  onQuickPrompt,
  onSubmit,
  selectedBenchmark
}: {
  activeRunId: string | null;
  busy: boolean;
  mainPrompt: string;
  onChange: (value: string) => void;
  onQuickPrompt: (text: string, options?: { mode?: RunMode; workspaceScope?: WorkspaceScope }) => void;
  onSubmit: (event: FormEvent) => void;
  selectedBenchmark: string;
}) {
  return (
    <section className="command-center" aria-label="ChatUI 实验入口">
      <div className="command-copy">
        <div className="command-orb">
          <Sparkles size={18} />
        </div>
        <div>
          <p className="eyebrow">ChatUI</p>
          <h2>欢迎来到 AgenticSciML</h2>
          <span>用自然语言启动实验、解释 trace、浏览 evidence，并打开 solution workspace。</span>
        </div>
      </div>
      <form className="main-composer" onSubmit={onSubmit}>
        <input
          value={mainPrompt}
          onChange={(event) => onChange(event.target.value)}
          placeholder="请输入你的实验意图，例如：跑 mock、解释这个 trace、打开 champion"
        />
        <div className="composer-tools">
          <button type="button" title="Artifact context">
            <Database size={15} />
          </button>
          <select aria-label="Agent">
            <option>Agent</option>
            <option>Trace Analyst</option>
            <option>Run Operator</option>
          </select>
          <button className="send-button" disabled={busy} type="submit" title="发送">
            <Send size={16} />
          </button>
        </div>
      </form>
      <div className="prompt-pills">
        <button type="button" onClick={() => onQuickPrompt(`跑一个 ${selectedBenchmark} mock 实验`, { mode: "mock" })}>
          跑 mock
        </button>
        <button type="button" disabled={!activeRunId} onClick={() => onQuickPrompt("解释这个 trace")}>
          解释 trace
        </button>
        <button
          type="button"
          disabled={!activeRunId}
          onClick={() => onQuickPrompt("打开 champion solution", { workspaceScope: "solution" })}
        >
          打开 champion
        </button>
        <button type="button" onClick={() => onQuickPrompt("比较两个 run")}>
          比较 run
        </button>
      </div>
    </section>
  );
}

function RunsView({
  activeRunId,
  runs,
  onRefresh,
  onSelectRun,
  onStartMock
}: {
  activeRunId: string | null;
  runs: RunSummary[];
  onRefresh: () => void;
  onSelectRun: (run: RunSummary) => void;
  onStartMock: () => void;
}) {
  return (
    <div className="view-stack">
      <section className="section-head">
        <div>
          <p className="eyebrow">Run catalog</p>
          <h2>实验运行记录</h2>
        </div>
        <div className="section-actions">
          <button type="button" onClick={onRefresh}>
            <RefreshCw size={15} />
            Refresh
          </button>
          <button type="button" onClick={onStartMock}>
            <Play size={15} />
            Mock
          </button>
        </div>
      </section>
      <DataRegion title="Runs">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>run</th>
                <th>status</th>
                <th>run_state</th>
                <th>champion</th>
                <th>solutions</th>
                <th>gate</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr
                  className={run.run_id === activeRunId ? "selected-row" : ""}
                  key={run.run_id}
                  onClick={() => onSelectRun(run)}
                >
                  <td>{run.run_id}</td>
                  <td>{run.status}</td>
                  <td>{run.metadata?.run_state ?? "unknown"}</td>
                  <td>{run.metadata?.champion_node_id ?? "none"}</td>
                  <td>{run.metadata?.solution_count ?? 0}</td>
                  <td>{String(run.trace_summary?.quality_gate?.passed ?? "pending")}</td>
                </tr>
              ))}
              {runs.length === 0 ? (
                <tr>
                  <td colSpan={6}>没有发现 run。可以先启动一个 mock run。</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </DataRegion>
    </div>
  );
}

function ArtifactsView({
  activeRun,
  artifactPayload,
  events,
  filter,
  selectedArtifactPath,
  onFilterChange,
  onSelectArtifact
}: {
  activeRun: RunSummary | null;
  artifactPayload: ArtifactPayload | null;
  events: string[];
  filter: string;
  selectedArtifactPath: string;
  onFilterChange: (value: string) => void;
  onSelectArtifact: (value: string) => void;
}) {
  return (
    <div className="artifact-workspace">
      <section className="artifact-browser">
        <div className="section-head compact">
          <div>
            <p className="eyebrow">Artifacts</p>
            <h2>{activeRun?.run_id ?? "未选择 run"}</h2>
          </div>
        </div>
        <div className="artifact-list-vertical">
          {(activeRun?.artifacts ?? []).map((artifact) => (
            <button
              className={selectedArtifactPath === artifact.path ? "artifact-row selected" : "artifact-row"}
              key={artifact.path}
              type="button"
              onClick={() => onSelectArtifact(artifact.path)}
            >
              {artifact.kind === "directory" ? <FolderTree size={14} /> : <FileText size={14} />}
              <span>{artifact.path}</span>
              <small>{artifact.kind}</small>
            </button>
          ))}
          {!activeRun ? <p className="muted">先在 Runs 或 Dashboard 中选择一个 run。</p> : null}
        </div>
      </section>
      <section className="artifact-detail">
        <DataRegion title="Artifact preview">
          <ArtifactPreview payload={artifactPayload} />
        </DataRegion>
        <DataRegion title="Trace events">
          <div className="filter-line">
            <input
              value={filter}
              onChange={(event) => onFilterChange(event.target.value)}
              placeholder="按 event name / node id / 文本过滤"
            />
          </div>
          <TraceList events={events} emptyText="没有匹配的 trace event。" />
        </DataRegion>
      </section>
    </div>
  );
}

function CodeView({
  activeRunId,
  codeServer,
  scope,
  onScopeChange
}: {
  activeRunId: string | null;
  codeServer: CodeServerPayload | null;
  scope: WorkspaceScope;
  onScopeChange: (scope: WorkspaceScope) => void;
}) {
  return (
    <div className="code-workspace">
      <section className="section-head">
        <div>
          <p className="eyebrow">VS Code Web</p>
          <h2>code-server sidecar</h2>
        </div>
        <label className="compact-field">
          <span>Workspace</span>
          <select value={scope} onChange={(event) => onScopeChange(event.target.value as WorkspaceScope)}>
            <option value="repo">repo</option>
            <option value="run">run</option>
            <option value="solution">solution</option>
          </select>
        </label>
      </section>
      <section className="vscode-frame">
        {codeServer ? (
          <iframe className="vscode-iframe" src={codeServer.url} title="VS Code Web" />
        ) : (
          <div className="vscode-empty">
            <TerminalSquare size={22} />
            <p>选择 active run 后可打开 run 或 solution workspace。</p>
          </div>
        )}
      </section>
      <section className="code-block">
        <div className="code-block-head">
          <TerminalSquare size={18} />
          <h3>Workspace</h3>
        </div>
        <div className="workspace-path">{codeServer?.workspace ?? "选择 active run 后可打开 run 或 solution workspace。"}</div>
        <a
          aria-disabled={!codeServer}
          className={codeServer ? "code-link" : "code-link disabled"}
          href={codeServer?.url ?? "#"}
          rel="noreferrer"
          target="_blank"
        >
          Open VS Code Web
          <ChevronRight size={15} />
        </a>
      </section>
      <section className="code-block compact">
        <div className="code-block-head">
          <ShieldCheck size={18} />
          <h3>Sidecar boundary</h3>
        </div>
        <pre>{codeServer?.command_hint ?? "PASSWORD=<local-token> code-server --bind-addr 127.0.0.1:8080 <workspace>"}</pre>
        <ul>
          {(codeServer?.warnings ?? [
            "code-server 必须单独启动在 127.0.0.1。",
            "ChatUI 不携带 token，也不自动绕过鉴权。"
          ]).map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
          {!activeRunId && scope !== "repo" ? <li>run / solution scope 需要先选择 active run。</li> : null}
        </ul>
      </section>
    </div>
  );
}

function AgentPanel({
  busy,
  collapsed,
  message,
  messages,
  mode,
  pendingRealAction,
  onChangeMessage,
  onDismissRealAction,
  onSend,
  onToggle
}: {
  busy: boolean;
  collapsed: boolean;
  message: string;
  messages: AgentMessage[];
  mode: RunMode;
  pendingRealAction: SolverAction | null;
  onChangeMessage: (value: string) => void;
  onDismissRealAction: () => void;
  onSend: (event: FormEvent) => void;
  onToggle: () => void;
}) {
  if (collapsed) {
    return (
      <aside className="agent-collapsed">
        <button aria-label="展开 Agent 面板" className="icon-button" type="button" onClick={onToggle}>
          <PanelRightOpen size={18} />
        </button>
        <Bot size={20} />
      </aside>
    );
  }
  return (
    <aside className="agent-panel" aria-label="Agent 面板">
      <header className="agent-header">
        <div>
          <p className="eyebrow">Agent</p>
          <h2>实验操作员</h2>
        </div>
        <button aria-label="收起 Agent 面板" className="icon-button" type="button" onClick={onToggle}>
          <PanelRightClose size={18} />
        </button>
      </header>
      <div className="agent-mode">
        <StatusBadge tone={mode === "real" ? "bad" : mode === "dry_run" ? "info" : "good"}>{mode}</StatusBadge>
        <span>受控 action 分发</span>
      </div>
      {pendingRealAction ? (
        <div className="pending-action">
          <AlertTriangle size={16} />
          <div>
            <strong>Real mode action 已拦截</strong>
            <p>真实 LLM 运行需要显式凭据、预算和 claim boundary 检查。当前不会自动执行。</p>
            <button type="button" onClick={onDismissRealAction}>
              Dismiss
            </button>
          </div>
        </div>
      ) : null}
      <div className="messages">
        {messages.map((item) => (
          <article className={`message ${item.role}`} key={item.id}>
            <span>{item.role}</span>
            <p>{item.text}</p>
            {item.response ? <StructuredResponse response={item.response} /> : null}
          </article>
        ))}
      </div>
      <form className="composer" onSubmit={onSend}>
        <input
          value={message}
          onChange={(event) => onChangeMessage(event.target.value)}
          placeholder="例如：跑 mock / 解释 trace / 打开 champion"
        />
        <button aria-label="发送" disabled={busy} type="submit">
          <Send size={15} />
        </button>
      </form>
    </aside>
  );
}

function ChatTranscript({ messages }: { messages: AgentMessage[] }) {
  return (
    <div className="chat-transcript">
      {messages.map((item) => (
        <article className={`message ${item.role}`} key={item.id}>
          <span>{item.role}</span>
          <p>{item.text}</p>
          {item.response ? <StructuredResponse response={item.response} /> : null}
        </article>
      ))}
    </div>
  );
}

function StructuredResponse({ response }: { response: SolverResponse }) {
  return (
    <div className="structured-response">
      {response.actions.length ? <KeyValueList label="actions" values={response.actions.map((action) => action.type)} /> : null}
      {response.warnings.length ? <KeyValueList label="warnings" values={response.warnings} tone="warning" /> : null}
      {response.artifacts.length ? <KeyValueList label="artifacts" values={response.artifacts.map((artifact) => String(artifact.path ?? "artifact"))} /> : null}
      {response.trace_refs.length ? <KeyValueList label="trace refs" values={response.trace_refs.map((ref) => JSON.stringify(ref))} /> : null}
    </div>
  );
}

function KeyValueList({ label, tone, values }: { label: string; tone?: "warning"; values: string[] }) {
  return (
    <div className={`kv-list ${tone ?? ""}`}>
      <strong>{label}</strong>
      {values.map((value) => (
        <span key={value}>{value}</span>
      ))}
    </div>
  );
}

function DataRegion({ children, title }: { children: ReactNode; title: string }) {
  return (
    <section className="data-region">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong title={value}>{value}</strong>
    </div>
  );
}

function StatusBadge({ children, tone }: { children: ReactNode; tone: "neutral" | "good" | "bad" | "info" }) {
  return <span className={`status-badge ${tone}`}>{children}</span>;
}

function Leaderboard({ rows }: { rows: Array<Record<string, string>> }) {
  return (
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
          {rows.slice(0, 8).map((row) => (
            <tr key={`${row.rank}-${row.node_id}`}>
              <td>{row.rank}</td>
              <td>{row.node_id}</td>
              <td>{row.metric}</td>
              <td>{row.score}</td>
              <td>{row.status}</td>
            </tr>
          ))}
          {rows.length === 0 ? (
            <tr>
              <td colSpan={5}>尚无 leaderboard。</td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  );
}

function ArtifactChips({ artifacts }: { artifacts: ArtifactEntry[] }) {
  if (!artifacts.length) return <p className="muted">尚无 artifact。</p>;
  return (
    <div className="artifact-chips">
      {artifacts.map((artifact) => (
        <span key={artifact.path}>{artifact.path}</span>
      ))}
    </div>
  );
}

function ArtifactPreview({ payload }: { payload: ArtifactPayload | null }) {
  if (!payload) return <p className="muted">选择左侧 artifact 后预览。</p>;
  if (payload.kind === "directory") {
    return (
      <div className="directory-preview">
        {payload.entries.map((entry) => (
          <div key={entry.path}>
            <span>{entry.kind}</span>
            <strong>{entry.path}</strong>
          </div>
        ))}
      </div>
    );
  }
  return <pre className="artifact-content">{payload.content}</pre>;
}

function TraceList({ emptyText, events }: { emptyText: string; events: string[] }) {
  if (!events.length) return <p className="muted">{emptyText}</p>;
  return (
    <div className="event-log">
      {events.map((event, index) => {
        const summary = traceSummary(event);
        return (
          <code key={`${index}-${event.slice(0, 24)}`}>
            <span>{summary}</span>
            {event}
          </code>
        );
      })}
    </div>
  );
}

function traceSummary(event: string) {
  try {
    const parsed = JSON.parse(event) as Record<string, unknown>;
    const name = parsed.name ?? parsed.event_name ?? parsed.event_type ?? "trace";
    const seq = parsed.event_seq ?? parsed.seq;
    return seq === undefined ? String(name) : `${String(name)} #${String(seq)}`;
  } catch {
    return "trace";
  }
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
