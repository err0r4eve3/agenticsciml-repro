import { FormEvent, ReactNode, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Bot,
  ChevronRight,
  Code2,
  Database,
  FileText,
  FolderTree,
  MessageSquare,
  PackageSearch,
  PanelRightClose,
  PanelRightOpen,
  Play,
  Plus,
  RefreshCw,
  Send,
  Sparkles,
  TerminalSquare,
  Users
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
type WorkspaceScope = "repo" | "account" | "run" | "solution";
type AssistantMode = "ask" | "plan" | "agent";
type ReasoningEffort = "low" | "medium" | "high";
type PageKey = "chat" | "ide" | "library";

type AccountOption = {
  account_id: string;
  display_name: string;
  workspace_root: string;
  runs_dir: string;
  isolation: string;
  auth: string;
};

type AlgorithmSpec = {
  id: string;
  name: string;
  family: string;
  compatible_benchmark_families: string[];
  benchmark_examples: string[];
  status: string;
  description: string;
  claim_boundary: string;
  safety_notes: string;
};

type AgentMessage = {
  id: number;
  role: "user" | "assistant";
  text: string;
  response?: SolverResponse;
};

type ModeModelSettings = {
  reasoning_effort: ReasoningEffort;
  temperature: number;
  source?: string;
};

type SolverSettings = {
  default_assistant_mode: AssistantMode;
  reasoning_efforts: ReasoningEffort[];
  temperature_range: [number, number];
  assistant_modes: Record<AssistantMode, ModeModelSettings>;
};

type SolverAction = {
  type: string;
  payload?: {
    benchmark?: string;
    mode?: RunMode;
    account_id?: string | null;
    background?: boolean;
    scope?: WorkspaceScope;
    [key: string]: unknown;
  };
  run_id?: string;
};

type SolverResponse = {
  assistant_mode: AssistantMode;
  model_settings: ModeModelSettings & { source: string };
  reply: string;
  actions: SolverAction[];
  artifacts: Array<Record<string, unknown>>;
  warnings: string[];
  trace_refs: Array<Record<string, unknown>>;
};

type CodeServerPayload = {
  scope: WorkspaceScope;
  account_id?: string | null;
  run_id?: string | null;
  solution_id?: string | null;
  workspace: string;
  url: string;
  warnings: string[];
  command_hint: string;
  isolation?: string;
};

type CodeWorkspaceOption = CodeServerPayload & {
  id: string;
  label: string;
  status: string;
};

const DEFAULT_SOLVER_SETTINGS: SolverSettings = {
  default_assistant_mode: "ask",
  reasoning_efforts: ["low", "medium", "high"],
  temperature_range: [0, 2],
  assistant_modes: {
    ask: { reasoning_effort: "medium", temperature: 0.2 },
    plan: { reasoning_effort: "high", temperature: 0.35 },
    agent: { reasoning_effort: "high", temperature: 0.1 }
  }
};

const api = {
  async getAccounts(): Promise<AccountOption[]> {
    const payload = await getJson<{ accounts: AccountOption[] }>("/api/accounts");
    return payload.accounts;
  },
  async createAccount(accountId: string, displayName?: string): Promise<AccountOption> {
    const payload = await postJson<{ account: AccountOption }>("/api/accounts", {
      account_id: accountId,
      display_name: displayName ?? accountId
    });
    return payload.account;
  },
  async getBenchmarks(): Promise<Benchmark[]> {
    const payload = await getJson<{ benchmarks: Benchmark[] }>("/api/benchmarks");
    return payload.benchmarks;
  },
  async getAlgorithms(): Promise<AlgorithmSpec[]> {
    const payload = await getJson<{ algorithms: AlgorithmSpec[] }>("/api/algorithms");
    return payload.algorithms;
  },
  async getSolverSettings(): Promise<SolverSettings> {
    return getJson<SolverSettings>("/api/solver/settings");
  },
  async getRuns(accountId: string): Promise<RunSummary[]> {
    const payload = await getJson<{ runs: RunSummary[] }>(`/api/runs?${accountParams(accountId)}`);
    return payload.runs;
  },
  async startRun(body: {
    benchmark: string;
    mode: RunMode;
    account_id: string;
    experiment_id?: string;
    max_iterations?: number;
    parallel_mutations?: number;
    background?: boolean;
  }): Promise<RunSummary> {
    const id = body.experiment_id ?? `web-${new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 14)}`;
    await postJson<unknown>("/api/runs", {
      benchmark: body.benchmark,
      mode: body.mode,
      account_id: body.account_id,
      experiment_id: id,
      max_iterations: body.max_iterations ?? 0,
      parallel_mutations: body.parallel_mutations ?? 1,
      background: body.background ?? false
    });
    return api.getRun(id, body.account_id);
  },
  async resumeRun(runId: string, benchmark: string, mode: RunMode, accountId: string): Promise<RunSummary> {
    await postJson<unknown>(`/api/runs/${encodeURIComponent(runId)}/resume`, {
      benchmark,
      mode,
      account_id: accountId,
      experiment_id: runId,
      max_iterations: 1,
      parallel_mutations: 1,
      background: true
    });
    return api.getRun(runId, accountId);
  },
  async getRun(runId: string, accountId: string): Promise<RunSummary> {
    return getJson<RunSummary>(`/api/runs/${encodeURIComponent(runId)}?${accountParams(accountId)}`);
  },
  async getArtifact(runId: string, artifactPath: string, accountId: string): Promise<ArtifactPayload> {
    return getJson<ArtifactPayload>(
      `/api/runs/${encodeURIComponent(runId)}/artifacts/${artifactPath
        .split("/")
        .map(encodeURIComponent)
        .join("/")}?${accountParams(accountId)}`
    );
  },
  async askSolver(body: {
    message: string;
    active_run_id: string | null;
    selected_benchmark: string;
    mode: RunMode;
    assistant_mode: AssistantMode;
    reasoning_effort?: ReasoningEffort;
    temperature?: number;
    workspace_scope: WorkspaceScope;
    account_id: string;
  }): Promise<SolverResponse> {
    return postJson<SolverResponse>("/api/solver/chat", body);
  },
  async getCodeWorkspaces(runId: string | null, accountId: string): Promise<CodeWorkspaceOption[]> {
    const params = new URLSearchParams();
    params.set("account_id", accountId);
    if (runId) params.set("run_id", runId);
    const payload = await getJson<{ workspaces: CodeWorkspaceOption[] }>(`/api/code-server/workspaces?${params.toString()}`);
    return payload.workspaces;
  }
};

export function App() {
  const [activePage, setActivePage] = useState<PageKey>("chat");
  const [accounts, setAccounts] = useState<AccountOption[]>([]);
  const [activeAccountId, setActiveAccountId] = useState("local");
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [algorithms, setAlgorithms] = useState<AlgorithmSpec[]>([]);
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
      text: "可以直接提问或描述任务。需要更多工具时，请使用左侧分页切换。"
    }
  ]);
  const [mode, setMode] = useState<RunMode>("mock");
  const [assistantMode, setAssistantMode] = useState<AssistantMode>("ask");
  const [solverSettings, setSolverSettings] = useState<SolverSettings>(DEFAULT_SOLVER_SETTINGS);
  const [workspaceScope, setWorkspaceScope] = useState<WorkspaceScope>("account");
  const [codeWorkspaces, setCodeWorkspaces] = useState<CodeWorkspaceOption[]>([]);
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState<string | null>(null);
  const [agentCollapsed, setAgentCollapsed] = useState(false);
  const [pendingRealAction, setPendingRealAction] = useState<SolverAction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getAccounts()
      .then((items) => {
        setAccounts(items);
        if (items.length && !items.some((item) => item.account_id === activeAccountId)) {
          setActiveAccountId(items[0].account_id);
        }
      })
      .catch((exc) => setError(String(exc)));
    api
      .getBenchmarks()
      .then((items) => {
        setBenchmarks(items);
        if (items.length && !items.some((item) => item.name === selectedBenchmark)) {
          setSelectedBenchmark(items[0].name);
        }
      })
      .catch((exc) => setError(String(exc)));
    api.getAlgorithms().then(setAlgorithms).catch((exc) => setError(String(exc)));
    api.getSolverSettings().then(setSolverSettings).catch((exc) => setError(String(exc)));
    refreshRuns().catch((exc) => setError(String(exc)));
  }, []);

  useEffect(() => {
    setActiveRun(null);
    setActiveRunId(null);
    setSelectedArtifactPath("");
    setSelectedWorkspaceId(null);
    setWorkspaceScope("account");
    refreshRuns().catch((exc) => setError(String(exc)));
    refreshCodeWorkspaces().catch((exc) => setError(String(exc)));
  }, [activeAccountId]);

  useEffect(() => {
    if (!activeRunId) return;
    refreshActiveRun(activeRunId).catch((exc) => setError(String(exc)));
  }, [activeRunId, activeAccountId]);

  useEffect(() => {
    if (!activeRunId) return;
    setEvents([]);
    const source = new EventSource(
      `/api/runs/${encodeURIComponent(activeRunId)}/events?follow=false&${accountParams(activeAccountId)}`
    );
    source.addEventListener("trace", (event) => {
      setEvents((current) => [event.data, ...current].slice(0, 80));
    });
    source.addEventListener("end", () => source.close());
    source.onerror = () => source.close();
    return () => source.close();
  }, [activeRunId, activeAccountId]);

  useEffect(() => {
    if (!activeRunId || !selectedArtifactPath) {
      setArtifactPayload(null);
      return;
    }
    api
      .getArtifact(activeRunId, selectedArtifactPath, activeAccountId)
      .then(setArtifactPayload)
      .catch((exc) => setError(String(exc)));
  }, [activeRunId, selectedArtifactPath, activeAccountId]);

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
  const selectedWorkspace = useMemo(
    () => codeWorkspaces.find((workspace) => workspace.id === selectedWorkspaceId) ?? null,
    [codeWorkspaces, selectedWorkspaceId]
  );

  useEffect(() => {
    refreshCodeWorkspaces().catch((exc) => setError(String(exc)));
  }, [activeRunId, runs.length, activeAccountId]);

  async function refreshRuns() {
    const payload = await api.getRuns(activeAccountId);
    setRuns(payload);
  }

  async function refreshActiveRun(runId = activeRunId) {
    if (!runId) return;
    const run = await api.getRun(runId, activeAccountId);
    setActiveRun(run);
    setRuns((current) => [run, ...current.filter((item) => item.run_id !== run.run_id)]);
  }

  async function refreshAll() {
    setError(null);
    await refreshRuns();
    await refreshActiveRun();
    await refreshCodeWorkspaces();
  }

  async function refreshCodeWorkspaces() {
    const workspaces = await api.getCodeWorkspaces(activeRunId, activeAccountId);
    setCodeWorkspaces(workspaces);
    setSelectedWorkspaceId((current) => (current && workspaces.some((workspace) => workspace.id === current) ? current : null));
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
        account_id: activeAccountId,
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

  async function submitAgentMessage(
    rawMessage: string,
    overrides: { mode?: RunMode; assistantMode?: AssistantMode; workspaceScope?: WorkspaceScope } = {}
  ) {
    if (!rawMessage.trim()) return;
    const userMessage = rawMessage.trim();
    setMessages((current) => [...current, { id: Date.now(), role: "user", text: userMessage }]);
    setBusy(true);
    setError(null);
    try {
      const requestWorkspaceScope = overrides.workspaceScope ?? workspaceScope;
      const requestAssistantMode = overrides.assistantMode ?? assistantMode;
      const response = await api.askSolver({
        message: userMessage,
        active_run_id: activeRunId,
        selected_benchmark: selectedBenchmark,
        mode: overrides.mode ?? mode,
        assistant_mode: requestAssistantMode,
        workspace_scope: requestWorkspaceScope,
        account_id: activeAccountId
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
      if (overrides.assistantMode && overrides.assistantMode !== assistantMode) setAssistantMode(overrides.assistantMode);
      if (requestAssistantMode === "agent") {
        await dispatchSolverActions(response.actions);
      }
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
        if (!actionBelongsToActiveAccount(action)) continue;
        if (actionMode === "real") {
          setPendingRealAction(action);
          continue;
        }
        await startRun(actionMode, Boolean(action.payload?.background));
      }
      if (action.type === "resume_run" && action.run_id) {
        if (!actionBelongsToActiveAccount(action)) continue;
        if (mode === "real") {
          setPendingRealAction(action);
          continue;
        }
        const run = await api.resumeRun(action.run_id, selectedBenchmark, mode, activeAccountId);
        setActiveRun(run);
        setActiveRunId(run.run_id);
        await refreshRuns();
      }
      if (action.type === "open_code_server") {
        const nextScope = action.payload?.scope;
        if (!actionBelongsToActiveAccount(action)) continue;
        if (nextScope === "repo") {
          addAssistantMessage("Agent 模式不能打开 shared repo；请使用当前账号 workspace。");
          continue;
        }
        if (nextScope === "account" || nextScope === "run" || nextScope === "solution") {
          setWorkspaceScope(nextScope);
        }
        setActivePage("ide");
        setSelectedWorkspaceId(null);
      }
      if (action.type === "summarize_artifact") {
        if (!actionBelongsToActiveAccount(action)) continue;
        const artifactPath = action.payload?.path;
        if (typeof artifactPath === "string") {
          setSelectedArtifactPath(artifactPath);
        }
        setActivePage("library");
      }
    }
  }

  function actionBelongsToActiveAccount(action: SolverAction) {
    const actionAccountId = action.payload?.account_id;
    if (typeof actionAccountId === "string" && actionAccountId !== activeAccountId) {
      addAssistantMessage("已拦截跨账号 action：Agent 只能操作当前账号创建的工作区内容。");
      return false;
    }
    return true;
  }

  function addAssistantMessage(text: string) {
    setMessages((current) => [...current, { id: Date.now(), role: "assistant", text }]);
  }

  function selectRun(run: RunSummary) {
    setActiveRun(run);
    setActiveRunId(run.run_id);
    setSelectedArtifactPath("");
  }

  function selectPage(page: PageKey) {
    setActivePage(page);
    if (page === "ide") {
      setSelectedWorkspaceId(null);
      refreshCodeWorkspaces().catch((exc) => setError(String(exc)));
    }
  }

  function openWorkspace(workspace: CodeWorkspaceOption) {
    setWorkspaceScope(workspace.scope);
    if (workspace.run_id) setActiveRunId(workspace.run_id);
    setSelectedWorkspaceId(workspace.id);
  }

  async function createLocalAccount() {
    const nextId = `user_${new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 12)}`;
    setBusy(true);
    setError(null);
    try {
      const account = await api.createAccount(nextId, nextId);
      setAccounts((current) => [...current.filter((item) => item.account_id !== account.account_id), account]);
      setActiveAccountId(account.account_id);
      setActivePage("ide");
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className={`workbench ${agentCollapsed ? "agent-is-collapsed" : ""}`}>
      <FunctionNav
        accounts={accounts}
        activeAccountId={activeAccountId}
        activePage={activePage}
        onAccountChange={setActiveAccountId}
        onCreateAccount={createLocalAccount}
        onSelectPage={selectPage}
      />
      {activePage === "chat" ? (
        <section className="chatgpt-page" aria-label="ChatUI 页面">
          <PageHeader title="AgenticSciML" subtitle="ChatUI" />
          {error ? <div className="error-line">{error}</div> : null}
          <PureChatUI
            assistantMode={assistantMode}
            modeSettings={solverSettings.assistant_modes[assistantMode]}
            busy={busy}
            mainPrompt={mainPrompt}
            messages={messages}
            onAssistantModeChange={setAssistantMode}
            onChange={setMainPrompt}
            onOpenIde={() => selectPage("ide")}
            onOpenLibrary={() => setActivePage("library")}
            onQuickPrompt={(text, options) => submitAgentMessage(text, options)}
            onSubmit={sendMainPrompt}
          />
        </section>
      ) : null}
      {activePage === "ide" ? (
        <section className={selectedWorkspace ? "ide-page workspace-open" : "ide-page workspace-select"} aria-label="AI IDE 页面">
          {selectedWorkspace ? (
            <>
              <CodeView codeServer={selectedWorkspace} />
              <AgentPanel
                assistantMode={assistantMode}
                modeSettings={solverSettings.assistant_modes[assistantMode]}
                collapsed={agentCollapsed}
                message={message}
                messages={messages}
                pendingRealAction={pendingRealAction}
                busy={busy}
                onAssistantModeChange={setAssistantMode}
                onChangeMessage={setMessage}
                onSend={sendMessage}
                onToggle={() => setAgentCollapsed((current) => !current)}
                onDismissRealAction={() => setPendingRealAction(null)}
              />
            </>
          ) : (
            <WorkspaceSelector
              activeRunId={activeRunId}
              accountId={activeAccountId}
              busy={busy}
              error={error}
              workspaces={codeWorkspaces}
              onCreateAccount={createLocalAccount}
              onOpenWorkspace={openWorkspace}
              onRefresh={refreshCodeWorkspaces}
            />
          )}
        </section>
      ) : null}
      {activePage === "library" ? (
        <section className="library-page" aria-label="算法库页面">
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
          <AlgorithmLibraryPage
            activeRun={activeRun}
            activeRunId={activeRunId}
            algorithms={algorithms}
            artifactPayload={artifactPayload}
            events={filteredEvents}
            filter={traceFilter}
            runs={runs}
            selected={selected}
            selectedArtifactPath={selectedArtifactPath}
            onFilterChange={setTraceFilter}
            onOpenArtifacts={() => setSelectedArtifactPath("trace_summary.json")}
            onOpenCode={() => setActivePage("ide")}
            onRefreshRuns={refreshRuns}
            onSelectArtifact={setSelectedArtifactPath}
            onSelectRun={selectRun}
            onStartMock={() => startRun("mock", false)}
          />
        </section>
      ) : null}
    </main>
  );
}

function FunctionNav({
  accounts,
  activeAccountId,
  activePage,
  onAccountChange,
  onCreateAccount,
  onSelectPage
}: {
  accounts: AccountOption[];
  activeAccountId: string;
  activePage: PageKey;
  onAccountChange: (accountId: string) => void;
  onCreateAccount: () => void;
  onSelectPage: (page: PageKey) => void;
}) {
  const pages: Array<{ key: PageKey; label: string; icon: ReactNode }> = [
    { key: "chat", label: "ChatUI", icon: <MessageSquare size={19} /> },
    { key: "ide", label: "VS Code", icon: <Code2 size={19} /> },
    { key: "library", label: "算法库", icon: <PackageSearch size={19} /> }
  ];

  return (
    <aside className="function-nav" aria-label="功能导航">
      <div className="nav-brand" title="AgenticSciML">
        AS
      </div>
      <nav className="nav-pages">
        {pages.map((page) => (
          <button
            aria-label={page.label}
            className={activePage === page.key ? "nav-page selected" : "nav-page"}
            key={page.key}
            onClick={() => onSelectPage(page.key)}
            title={page.label}
            type="button"
          >
            {page.icon}
            <span>{page.label}</span>
          </button>
        ))}
      </nav>
      <div className="nav-account" title="本地账号空间">
        <Users size={15} />
        <select
          aria-label="本地账号空间"
          value={activeAccountId}
          onChange={(event) => onAccountChange(event.target.value)}
        >
          {accounts.map((account) => (
            <option key={account.account_id} value={account.account_id}>
              {account.display_name}
            </option>
          ))}
        </select>
        <button aria-label="新建账号空间" type="button" onClick={onCreateAccount}>
          <Plus size={14} />
        </button>
      </div>
    </aside>
  );
}

function PageHeader({ subtitle, title }: { subtitle: string; title: string }) {
  return (
    <header className="page-header">
      <div>
        <h1>{title}</h1>
        <span>{subtitle}</span>
      </div>
    </header>
  );
}

function PureChatUI({
  assistantMode,
  modeSettings,
  busy,
  mainPrompt,
  messages,
  onAssistantModeChange,
  onChange,
  onOpenIde,
  onOpenLibrary,
  onQuickPrompt,
  onSubmit
}: {
  assistantMode: AssistantMode;
  modeSettings: ModeModelSettings;
  busy: boolean;
  mainPrompt: string;
  messages: AgentMessage[];
  onAssistantModeChange: (mode: AssistantMode) => void;
  onChange: (value: string) => void;
  onOpenIde: () => void;
  onOpenLibrary: () => void;
  onQuickPrompt: (text: string, options?: { mode?: RunMode; assistantMode?: AssistantMode; workspaceScope?: WorkspaceScope }) => void;
  onSubmit: (event: FormEvent) => void;
}) {
  return (
    <section className="pure-chat">
      <div className="pure-chat-scroll">
        <div className="pure-chat-thread">
          <div className="chat-welcome">
            <div className="command-orb">
              <Sparkles size={18} />
            </div>
            <h2>今天要做什么？</h2>
            <p>输入问题或任务；需要更多工具时，使用左侧分页切换。</p>
            <div className="prompt-pills">
              <button type="button" onClick={() => onQuickPrompt("介绍一下这个项目当前能做什么")}>
                介绍项目
              </button>
              <button type="button" onClick={() => onQuickPrompt("帮我规划下一步", { assistantMode: "plan" })}>
                规划任务
              </button>
              <button type="button" onClick={onOpenLibrary}>
                打开算法库
              </button>
              <button type="button" onClick={onOpenIde}>
                打开 VS Code Web
              </button>
            </div>
          </div>
          <ChatTranscript messages={messages} />
        </div>
      </div>
      <form className="pure-composer" onSubmit={onSubmit}>
        <input
          value={mainPrompt}
          onChange={(event) => onChange(event.target.value)}
          placeholder="给 AgenticSciML 发消息"
        />
        <div className="composer-tools">
          <AssistantModeSwitch mode={assistantMode} settings={modeSettings} onChange={onAssistantModeChange} />
          <button type="button" title="Artifact context">
            <Database size={15} />
          </button>
          <button className="send-button" disabled={busy} type="submit" title="发送">
            <Send size={16} />
          </button>
        </div>
      </form>
    </section>
  );
}

function AssistantModeSwitch({
  mode,
  onChange,
  settings,
  compact = false
}: {
  mode: AssistantMode;
  onChange: (mode: AssistantMode) => void;
  settings?: ModeModelSettings;
  compact?: boolean;
}) {
  return (
    <div className={`assistant-mode-wrap ${compact ? "compact" : ""}`}>
      <div className={`assistant-mode ${compact ? "compact" : ""}`} aria-label="Assistant mode">
        {(["ask", "plan", "agent"] as const).map((item) => (
          <button
            key={item}
            className={mode === item ? "selected" : ""}
            type="button"
            onClick={() => onChange(item)}
          >
            {item}
          </button>
        ))}
      </div>
      {settings ? (
        <span className="mode-setting" title={`reasoning_effort=${settings.reasoning_effort}; temperature=${settings.temperature}`}>
          think {settings.reasoning_effort} · temp {settings.temperature}
        </span>
      ) : null}
    </div>
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
        <h1>算法库</h1>
        <span>实验工作台</span>
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

function AlgorithmLibraryPage({
  activeRun,
  activeRunId,
  algorithms,
  artifactPayload,
  events,
  filter,
  runs,
  selected,
  selectedArtifactPath,
  onFilterChange,
  onOpenArtifacts,
  onOpenCode,
  onRefreshRuns,
  onSelectArtifact,
  onSelectRun,
  onStartMock
}: {
  activeRun: RunSummary | null;
  activeRunId: string | null;
  algorithms: AlgorithmSpec[];
  artifactPayload: ArtifactPayload | null;
  events: string[];
  filter: string;
  runs: RunSummary[];
  selected?: Benchmark;
  selectedArtifactPath: string;
  onFilterChange: (value: string) => void;
  onOpenArtifacts: () => void;
  onOpenCode: () => void;
  onRefreshRuns: () => void;
  onSelectArtifact: (value: string) => void;
  onSelectRun: (run: RunSummary) => void;
  onStartMock: () => void;
}) {
  return (
    <section className="library-content">
      <section className="library-hero">
        <div>
          <p className="eyebrow">Algorithm Library</p>
          <h2>算法库</h2>
          <span>集中管理 benchmark、run、leaderboard、trace 和 artifact。ChatUI 与 VS Code Web 页面保持轻量。</span>
        </div>
      </section>
      <AlgorithmCatalog algorithms={algorithms} />
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

function AlgorithmCatalog({ algorithms }: { algorithms: AlgorithmSpec[] }) {
  const visible = algorithms;
  return (
    <DataRegion title="Algorithm catalog">
      <div className="algorithm-grid">
        {visible.map((algorithm) => (
          <article className="algorithm-card" key={algorithm.id}>
            <div>
              <span>{algorithm.family}</span>
              <h3>{algorithm.name}</h3>
            </div>
            <p>{algorithm.description}</p>
            <div className="algorithm-meta">
              <small>{algorithm.status}</small>
              <small>{algorithm.compatible_benchmark_families.join(" / ")}</small>
            </div>
            <strong>{algorithm.safety_notes}</strong>
          </article>
        ))}
        {visible.length === 0 ? <p className="muted">算法目录暂未加载。</p> : null}
      </div>
    </DataRegion>
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
  assistantMode,
  busy,
  mainPrompt,
  onAssistantModeChange,
  onChange,
  onQuickPrompt,
  onSubmit,
  selectedBenchmark
}: {
  activeRunId: string | null;
  assistantMode: AssistantMode;
  busy: boolean;
  mainPrompt: string;
  onAssistantModeChange: (mode: AssistantMode) => void;
  onChange: (value: string) => void;
  onQuickPrompt: (text: string, options?: { mode?: RunMode; assistantMode?: AssistantMode; workspaceScope?: WorkspaceScope }) => void;
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
          <AssistantModeSwitch compact mode={assistantMode} onChange={onAssistantModeChange} />
          <button className="send-button" disabled={busy} type="submit" title="发送">
            <Send size={16} />
          </button>
        </div>
      </form>
      <div className="prompt-pills">
        <button
          type="button"
          onClick={() => onQuickPrompt(`跑一个 ${selectedBenchmark} mock 实验`, { mode: "mock", assistantMode: "agent" })}
        >
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
  codeServer
}: {
  codeServer: CodeServerPayload | null;
}) {
  return (
    <section className="vscode-frame ide-vscode-frame">
      {codeServer ? (
        <iframe className="vscode-iframe" src={codeServer.url} title="VS Code Web" />
      ) : (
        <div className="vscode-empty">
          <TerminalSquare size={22} />
          <p>请选择工作空间。</p>
        </div>
      )}
    </section>
  );
}

function WorkspaceSelector({
  accountId,
  activeRunId,
  busy,
  error,
  workspaces,
  onCreateAccount,
  onOpenWorkspace,
  onRefresh
}: {
  accountId: string;
  activeRunId: string | null;
  busy: boolean;
  error: string | null;
  workspaces: CodeWorkspaceOption[];
  onCreateAccount: () => void;
  onOpenWorkspace: (workspace: CodeWorkspaceOption) => void;
  onRefresh: () => void;
}) {
  return (
    <section className="workspace-selector" aria-label="工作空间选择">
      <header className="workspace-selector-head">
        <div>
          <h1>工作空间</h1>
          <span>
            {activeRunId
              ? `account: ${accountId} / active run: ${activeRunId}`
              : `account: ${accountId} / 选择一个独立代码目录进入 AI IDE`}
          </span>
        </div>
        <div className="workspace-actions">
          <button className="icon-text-button" disabled={busy} type="button" onClick={onCreateAccount}>
            <Plus size={15} />
            新建账号空间
          </button>
          <button className="icon-text-button" disabled={busy} type="button" onClick={onRefresh}>
            <RefreshCw size={15} />
            刷新
          </button>
        </div>
      </header>
      {error ? <div className="error-line inline">{error}</div> : null}
      <div className="workspace-table">
        <table>
          <thead>
            <tr>
              <th>名称</th>
              <th>账号</th>
              <th>类型</th>
              <th>代码目录</th>
              <th>状态</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {workspaces.map((workspace) => (
              <tr key={workspace.id} onClick={() => onOpenWorkspace(workspace)}>
                <td>{workspace.label}</td>
                <td>{workspace.account_id ?? "shared"}</td>
                <td>{workspace.scope}</td>
                <td>{workspace.workspace}</td>
                <td>{workspace.status}</td>
                <td>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      onOpenWorkspace(workspace);
                    }}
                  >
                    打开
                    <ChevronRight size={15} />
                  </button>
                </td>
              </tr>
            ))}
            {workspaces.length === 0 ? (
              <tr>
                <td colSpan={6}>暂无可用工作空间。</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function AgentPanel({
  assistantMode,
  modeSettings,
  busy,
  collapsed,
  message,
  messages,
  pendingRealAction,
  onAssistantModeChange,
  onChangeMessage,
  onDismissRealAction,
  onSend,
  onToggle
}: {
  assistantMode: AssistantMode;
  modeSettings: ModeModelSettings;
  busy: boolean;
  collapsed: boolean;
  message: string;
  messages: AgentMessage[];
  pendingRealAction: SolverAction | null;
  onAssistantModeChange: (mode: AssistantMode) => void;
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
          <p className="eyebrow">AI</p>
          <h2>随时发问</h2>
        </div>
        <button aria-label="收起 Agent 面板" className="icon-button" type="button" onClick={onToggle}>
          <PanelRightClose size={18} />
        </button>
      </header>
      <AssistantModeSwitch compact mode={assistantMode} settings={modeSettings} onChange={onAssistantModeChange} />
      {pendingRealAction ? (
        <div className="pending-action">
          <AlertTriangle size={16} />
          <div>
            <strong>Real mode action 已拦截</strong>
            <p>真实模型动作需要显式凭据、预算和边界检查。当前不会自动执行。</p>
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
          placeholder="请输入你的问题"
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
  const settings = [
    `thinking ${response.model_settings.reasoning_effort}`,
    `temp ${response.model_settings.temperature}`,
  ];
  return (
    <div className="structured-response">
      <KeyValueList label="mode" values={[response.assistant_mode]} />
      <KeyValueList label="settings" values={settings} />
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

function accountParams(accountId: string) {
  const params = new URLSearchParams();
  params.set("account_id", accountId);
  return params.toString();
}
