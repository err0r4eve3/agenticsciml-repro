import {
  type CSSProperties,
  type FormEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useState
} from "react";
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
  Users,
  X
} from "lucide-react";

type Benchmark = {
  name: string;
  path: string;
  paper_section: string;
  paper_task_name?: string;
  family: string;
  metric: string;
  description?: string;
  fidelity_level: string;
  paper_gap_notes?: string;
  claim_boundaries?: string[];
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
    claim_gate?: ClaimGate;
    kb_manifest?: KbManifest;
    multimodal_evidence?: MultimodalEvidence;
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

type ClaimGate = {
  schema_version: number;
  claim_level: ClaimLevel;
  status: "allowed" | "blocked" | "downgraded";
  paper_level_claim_supported: boolean;
  scientific_claim_supported: boolean;
  evaluator_trust_level: string;
  paper_benchmark_equivalent: boolean;
  domain_evaluator_present: boolean;
  metric_validated_by_domain_expert: boolean;
  reasons: string[];
};

type KbManifest = {
  coverage_status: string;
  paper_kb_equivalent: boolean;
  entry_count: number;
  paper_reference_entry_count: number;
};

type MultimodalEvidence = {
  actual_image_inputs_used: boolean;
  analysis_mode: string;
  plot_artifact_generated?: boolean;
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
type ClaimLevel = "workflow_proxy" | "paper_workflow";
type WorkspaceScope = "repo" | "account" | "run" | "solution";
type AssistantMode = "ask" | "plan" | "agent";
type ReasoningEffort = "low" | "medium" | "high" | "xhigh";
type PageKey = "chat" | "ide" | "library";

const AGENT_PANEL_MIN_WIDTH = 280;
const AGENT_PANEL_MAX_WIDTH = 560;
const AGENT_PANEL_DEFAULT_WIDTH = 360;

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
  description_zh?: string;
  features?: string[];
  features_zh?: string[];
  problem_fit?: string[];
  problem_fit_zh?: string[];
  claim_boundary: string;
  safety_notes: string;
  safety_notes_zh?: string;
  source_scope?: string;
  implementation_path?: string | null;
  operator?: {
    scheduler_mode?: string;
    default_axis?: string;
    mutation_axes?: string[];
    expected_static_terms?: string[];
    claim_boundary?: string;
  };
};

type LibraryLanguage = "zh" | "en";

type PaperTask = {
  paper_section: string;
  title: string;
  summary: string;
  benchmarks: Benchmark[];
  algorithms: AlgorithmSpec[];
  reference_primitives: string[];
  figure_labels: string[];
  local_artifact_figures: string[];
  claim_boundary: string;
};

type SelectorVotesPayload = {
  run_id: string;
  available: boolean;
  path: string;
  selected_parent_ids: string[];
  vote_counts: Record<string, number>;
  votes: Array<Record<string, unknown>>;
};

type SolutionSummary = {
  node_id: string;
  parent_id?: string | null;
  children: string[];
  status?: string;
  metric?: string | null;
  score?: number | null;
  loss?: number | null;
  higher_is_better?: boolean | null;
  score_delta_from_parent?: number | null;
  method_tags: string[];
  failure_kind?: string | null;
  num_debug_attempts?: number | null;
  emergence_audit?: {
    available: boolean;
    claim_level?: string | null;
    blocking_gap_count?: number;
  };
  kb_application?: {
    available: boolean;
    status?: string | null;
    retrieved_entry_id?: string | null;
    warning_count?: number;
    implemented_count?: number;
  };
  mutation_effect?: {
    available: boolean;
    status?: string | null;
    code_changed_from_parent?: boolean | null;
    duplicate_of?: string | null;
    diff_line_count?: number | null;
    operator_id?: string | null;
    mutation_axis?: string | null;
  };
  operator_assignment?: {
    available: boolean;
    scheduler_mode?: string | null;
    operator_id?: string | null;
    operator_name?: string | null;
    mutation_axis?: string | null;
    selection_source?: string | null;
    expected_term_count?: number;
    warning_count?: number;
  };
  workspace: string;
  artifacts: ArtifactEntry[];
};

type SolutionsPayload = {
  run_id: string;
  available: boolean;
  tree: {
    root_id?: string | null;
    node_count: number;
    schema_version?: number | null;
  };
  solutions: SolutionSummary[];
  leaderboard: Array<Record<string, string>>;
  evolution_health?: {
    unique_code_count?: number;
    duplicate_code_count?: number;
    max_plateau_length?: number;
    best_improvement?: number | null;
    operator_scheduler_mode?: string;
    operator_assignment_count?: number;
    operator_assignment_expected_count?: number;
    missing_operator_assignment_nodes?: string[];
    operator_method_tag_mismatch_nodes?: string[];
    operator_assignment_warning_count?: number;
    operator_health?: Record<string, {
      assigned?: number;
      evaluated?: number;
      duplicate?: number;
      plateau?: number;
      improved?: number;
      best_improvement?: number | null;
      axes?: Record<string, number>;
    }>;
    warnings?: string[];
  };
  innovation_report?: {
    available: boolean;
    innovation_claim_level?: string | null;
    scientific_novelty_supported?: boolean | null;
    paper_level_discovery_supported?: boolean | null;
    novelty_axis_count?: number | null;
    candidate_emergent_count?: number | null;
    operator_count?: number | null;
    warning_count?: number | null;
    top_axes?: Array<string | null>;
    claim_boundary?: string | null;
  };
  figures: ArtifactEntry[];
};

type RunConfig = {
  target_solution_count: number;
  max_iterations: number;
  parallel_mutations: number;
  selector_vote_count: number;
  max_children_per_node: number;
};

type StrategyLock = {
  lock_id: string;
  kind: "constraint" | "invariant" | "mathematical_intuition" | "modeling_choice" | "assumption";
  text: string;
  scope: "all_branches" | "selected_algorithms";
  required: boolean;
};

type ReadinessCheck = {
  check_id: string;
  category: string;
  severity: "blocker" | "warning" | "info";
  passed: boolean;
  message: string;
};

type ReadinessReport = {
  readiness_id: string;
  status: "ready" | "ready_with_warnings" | "blocked";
  launch_allowed: boolean;
  summary: {
    blocker_count: number;
    warning_count: number;
    info_count: number;
    check_count: number;
  };
  checks: ReadinessCheck[];
  claim_gate: ClaimGate;
  kb_manifest?: KbManifest;
  selector_panel_preview?: {
    configured_member_count: number;
    configured_heterogeneous: boolean;
  };
  benchmark_fidelity_preview: Array<{
    benchmark: string;
    fidelity_level: string;
    disallowed_claims: string[];
  }>;
  algorithm_seed_preview: Array<{
    algorithm_id: string;
    catalog_role: string;
    is_evaluated_implementation: boolean;
  }>;
  claim_boundary: string;
};

type AgentRole = {
  role: string;
  label: string;
  kind: string;
  default_model_settings?: {
    temperature: number;
    reasoning_effort: ReasoningEffort;
    rationale?: string;
  };
};

type AgentModelConfig = {
  model: string;
  temperature: number;
  reasoning_effort?: ReasoningEffort;
};

type ProblemIntakeState = {
  problem_statement: string;
  requirements: string;
  evaluation_criteria: string;
  data_description: string;
};

type ProblemRunPlan = {
  problem_summary: string;
  problem_intake: Record<string, unknown>;
  planner_snapshot: Record<string, unknown>;
  recommended_benchmark: Benchmark;
  benchmark_candidates: Array<{ benchmark: Benchmark; score: number; rationale: string }>;
  algorithm_rankings: Array<{
    algorithm: AlgorithmSpec;
    score: number;
    selected: boolean;
    source: string;
    rationale: string;
  }>;
  selected_algorithm_ids: string[];
  run_config: RunConfig & { mode: RunMode; planned_solution_budget: number };
  actions: SolverAction[];
  warnings: string[];
  claim_boundary: string;
  custom_problem_package?: null | {
    status: string;
    synthesis_level: string;
    evaluator_trust_level: string;
    paper_benchmark_equivalent: boolean;
    domain_evaluator_present: boolean;
    metric_validated_by_domain_expert: boolean;
    requires_replacement_for_scientific_claim: boolean;
    claim_boundary: string;
  };
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
  agent_role_defaults?: Record<
    string,
    {
      temperature: number;
      reasoning_effort: ReasoningEffort;
      rationale?: string;
    }
  >;
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
  auth_mode?: string;
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
  reasoning_efforts: ["low", "medium", "high", "xhigh"],
  temperature_range: [0, 2],
  assistant_modes: {
    ask: { reasoning_effort: "medium", temperature: 0.2 },
    plan: { reasoning_effort: "high", temperature: 0.35 },
    agent: { reasoning_effort: "high", temperature: 0.1 }
  }
};

const FALLBACK_AGENT_MODEL_SETTINGS = {
  temperature: 0,
  reasoning_effort: "medium" as ReasoningEffort
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
  async getPaperTasks(): Promise<PaperTask[]> {
    const payload = await getJson<{ tasks: PaperTask[] }>("/api/paper-tasks");
    return payload.tasks;
  },
  async getAgentRoles(): Promise<AgentRole[]> {
    const payload = await getJson<{ roles: AgentRole[] }>("/api/agent-roles");
    return payload.roles;
  },
  async planProblem(body: ProblemIntakeState & {
    mode: RunMode;
    target_solution_count: number;
    parallel_mutations: number;
    selector_vote_count: number;
    max_children_per_node: number;
    selected_algorithm_ids: string[];
    agent_models: Record<string, AgentModelConfig>;
    claim_level?: ClaimLevel;
  }): Promise<ProblemRunPlan> {
    return postJson<ProblemRunPlan>("/api/problem-intake/plan", body);
  },
  async previewReadiness(body: {
    benchmark: string;
    mode: RunMode;
    account_id: string;
    target_solution_count: number;
    max_iterations: number;
    parallel_mutations: number;
    selector_vote_count: number;
    max_children_per_node: number;
    selected_algorithm_ids: string[];
    manual_strategy_locks?: StrategyLock[];
    branch_context?: Record<string, unknown>;
    problem_intake?: Record<string, unknown>;
    planner_snapshot?: Record<string, unknown>;
    claim_level?: ClaimLevel;
    domain_evaluator_approved?: boolean;
    domain_reviewer?: string | null;
    domain_review_notes?: string | null;
    paper_benchmark_approved?: boolean;
    real_confirmed?: boolean;
  }): Promise<ReadinessReport> {
    return postJson<ReadinessReport>("/api/run-readiness/preview", body);
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
    target_solution_count?: number;
    max_iterations?: number;
    parallel_mutations?: number;
    selector_vote_count?: number;
    max_children_per_node?: number;
    agent_models?: Record<string, AgentModelConfig>;
    selected_algorithm_ids?: string[];
    manual_strategy_locks?: StrategyLock[];
    branch_context?: Record<string, unknown>;
    problem_intake?: Record<string, unknown>;
    planner_snapshot?: Record<string, unknown>;
    claim_level?: ClaimLevel;
    domain_evaluator_approved?: boolean;
    domain_reviewer?: string | null;
    domain_review_notes?: string | null;
    paper_benchmark_approved?: boolean;
    background?: boolean;
    real_confirmed?: boolean;
  }): Promise<RunSummary> {
    const id = body.experiment_id ?? `web-${new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 14)}`;
    await postJson<unknown>("/api/runs", {
      benchmark: body.benchmark,
      mode: body.mode,
      account_id: body.account_id,
      experiment_id: id,
      target_solution_count: body.target_solution_count,
      max_iterations: body.max_iterations ?? 0,
      parallel_mutations: body.parallel_mutations ?? 1,
      selector_vote_count: body.selector_vote_count ?? 3,
      max_children_per_node: body.max_children_per_node ?? 10,
      agent_models: body.agent_models ?? {},
      selected_algorithm_ids: body.selected_algorithm_ids ?? [],
      manual_strategy_locks: body.manual_strategy_locks ?? [],
      branch_context: body.branch_context ?? {},
      problem_intake: body.problem_intake ?? {},
      planner_snapshot: body.planner_snapshot ?? {},
      claim_level: body.claim_level ?? "workflow_proxy",
      domain_evaluator_approved: body.domain_evaluator_approved ?? false,
      domain_reviewer: body.domain_reviewer ?? null,
      domain_review_notes: body.domain_review_notes ?? null,
      paper_benchmark_approved: body.paper_benchmark_approved ?? false,
      background: body.background ?? false,
      real_confirmed: body.real_confirmed ?? false
    });
    return api.getRun(id, body.account_id);
  },
  async resumeRun(
    runId: string,
    benchmark: string,
    mode: RunMode,
    accountId: string,
    realConfirmed = false
  ): Promise<RunSummary> {
    await postJson<unknown>(`/api/runs/${encodeURIComponent(runId)}/resume`, {
      mode,
      account_id: accountId,
      experiment_id: runId,
      max_iterations: 1,
      parallel_mutations: 1,
      background: true,
      real_confirmed: realConfirmed
    });
    return api.getRun(runId, accountId);
  },
  async getRun(runId: string, accountId: string): Promise<RunSummary> {
    return getJson<RunSummary>(`/api/runs/${encodeURIComponent(runId)}?${accountParams(accountId)}`);
  },
  async getSelectorVotes(runId: string, accountId: string): Promise<SelectorVotesPayload> {
    return getJson<SelectorVotesPayload>(
      `/api/runs/${encodeURIComponent(runId)}/selector-votes?${accountParams(accountId)}`
    );
  },
  async getSolutions(runId: string, accountId: string): Promise<SolutionsPayload> {
    return getJson<SolutionsPayload>(
      `/api/runs/${encodeURIComponent(runId)}/solutions?${accountParams(accountId)}`
    );
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
    target_solution_count: number;
    parallel_mutations: number;
    selector_vote_count: number;
    max_children_per_node: number;
    selected_algorithm_ids: string[];
    agent_models: Record<string, AgentModelConfig>;
    claim_level?: ClaimLevel;
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
  const [libraryLanguage, setLibraryLanguage] = useState<LibraryLanguage>("zh");
  const [paperTasks, setPaperTasks] = useState<PaperTask[]>([]);
  const [agentRoles, setAgentRoles] = useState<AgentRole[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedBenchmark, setSelectedBenchmark] = useState("function_approx");
  const [selectedPaperSection, setSelectedPaperSection] = useState("S1.1");
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [activeRun, setActiveRun] = useState<RunSummary | null>(null);
  const [selectorVotes, setSelectorVotes] = useState<SelectorVotesPayload | null>(null);
  const [solutionsPayload, setSolutionsPayload] = useState<SolutionsPayload | null>(null);
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
  const [runConfig, setRunConfig] = useState<RunConfig>({
    target_solution_count: 3,
    max_iterations: 2,
    parallel_mutations: 1,
    selector_vote_count: 3,
    max_children_per_node: 10
  });
  const [agentModels, setAgentModels] = useState<Record<string, AgentModelConfig>>({});
  const [selectedAlgorithmIds, setSelectedAlgorithmIds] = useState<string[]>([]);
  const [strategyLocks, setStrategyLocks] = useState<StrategyLock[]>([]);
  const [problemIntake, setProblemIntake] = useState<ProblemIntakeState>({
    problem_statement: "",
    requirements: "",
    evaluation_criteria: "",
    data_description: ""
  });
  const [problemPlan, setProblemPlan] = useState<ProblemRunPlan | null>(null);
  const [readinessReport, setReadinessReport] = useState<ReadinessReport | null>(null);
  const [assistantMode, setAssistantMode] = useState<AssistantMode>("ask");
  const [solverSettings, setSolverSettings] = useState<SolverSettings>(DEFAULT_SOLVER_SETTINGS);
  const [workspaceScope, setWorkspaceScope] = useState<WorkspaceScope>("account");
  const [codeWorkspaces, setCodeWorkspaces] = useState<CodeWorkspaceOption[]>([]);
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState<string | null>(null);
  const [agentCollapsed, setAgentCollapsed] = useState(false);
  const [agentPanelWidth, setAgentPanelWidth] = useState(AGENT_PANEL_DEFAULT_WIDTH);
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
    api.getAgentRoles().then(setAgentRoles).catch((exc) => setError(String(exc)));
    api
      .getPaperTasks()
      .then((items) => {
        setPaperTasks(items);
        if (items.length && !items.some((item) => item.paper_section === selectedPaperSection)) {
          setSelectedPaperSection(items[0].paper_section);
        }
      })
      .catch((exc) => setError(String(exc)));
    api.getSolverSettings().then(setSolverSettings).catch((exc) => setError(String(exc)));
    refreshRuns().catch((exc) => setError(String(exc)));
  }, []);

  useEffect(() => {
    setActiveRun(null);
    setActiveRunId(null);
    setSelectorVotes(null);
    setSolutionsPayload(null);
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
    if (!activeRunId) {
      setSelectorVotes(null);
      setSolutionsPayload(null);
      return;
    }
    refreshRunEvidence(activeRunId).catch((exc) => setError(String(exc)));
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
  const selectedPaperTask = useMemo(
    () => paperTasks.find((task) => task.paper_section === selectedPaperSection) ?? paperTasks[0] ?? null,
    [paperTasks, selectedPaperSection]
  );
  const runBudgetPreview = useMemo(() => {
    const maxIterations =
      runConfig.target_solution_count > 0
        ? Math.ceil(Math.max(0, runConfig.target_solution_count - 1) / Math.max(1, runConfig.parallel_mutations))
        : runConfig.max_iterations;
    return {
      max_iterations: maxIterations,
      planned_solution_budget: 1 + maxIterations * runConfig.parallel_mutations
    };
  }, [runConfig]);
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

  async function refreshRunEvidence(runId = activeRunId) {
    if (!runId) return;
    await refreshRunEvidenceForAccount(runId, activeAccountId);
  }

  async function refreshRunEvidenceForAccount(runId: string, accountId: string) {
    const [votes, solutions] = await Promise.all([
      api.getSelectorVotes(runId, accountId),
      api.getSolutions(runId, accountId)
    ]);
    setSelectorVotes(votes);
    setSolutionsPayload(solutions);
  }

  function pollBackgroundRun(runId: string, accountId = activeAccountId, attemptsLeft = 8) {
    window.setTimeout(async () => {
      try {
        const run = await api.getRun(runId, accountId);
        setActiveRun((current) => (!current || current.run_id === runId ? run : current));
        setRuns((current) => [run, ...current.filter((item) => item.run_id !== run.run_id)]);
        if (run.status !== "running" || attemptsLeft <= 1) {
          await refreshRunEvidenceForAccount(runId, accountId);
          const workspaces = await api.getCodeWorkspaces(runId, accountId);
          setCodeWorkspaces(workspaces);
          setSelectedWorkspaceId((current) =>
            current && workspaces.some((workspace) => workspace.id === current) ? current : null
          );
          return;
        }
        pollBackgroundRun(runId, accountId, attemptsLeft - 1);
      } catch (exc) {
        setError(String(exc));
      }
    }, attemptsLeft === 8 ? 1200 : 1000);
  }

  async function refreshAll() {
    setError(null);
    await refreshRuns();
    await refreshActiveRun();
    await refreshRunEvidence();
    await refreshCodeWorkspaces();
  }

  async function refreshCodeWorkspaces() {
    const workspaces = await api.getCodeWorkspaces(activeRunId, activeAccountId);
    setCodeWorkspaces(workspaces);
    setSelectedWorkspaceId((current) => (current && workspaces.some((workspace) => workspace.id === current) ? current : null));
  }

  async function selectWorkspaceFromCodeServerAction(action: SolverAction) {
    const payload = action.payload ?? {};
    const actionRunId = typeof payload.run_id === "string" ? payload.run_id : activeRunId;
    const workspaces = await api.getCodeWorkspaces(actionRunId, activeAccountId);
    setCodeWorkspaces(workspaces);

    const targetWorkspace = typeof payload.workspace === "string" ? payload.workspace : null;
    const targetScope = payload.scope;
    const targetSolutionId = typeof payload.solution_id === "string" ? payload.solution_id : null;
    const match =
      workspaces.find((workspace) => targetWorkspace && workspace.workspace === targetWorkspace) ??
      workspaces.find((workspace) => {
        if (targetScope && workspace.scope !== targetScope) return false;
        if (actionRunId && workspace.run_id !== actionRunId) return false;
        if (targetSolutionId && workspace.solution_id !== targetSolutionId) return false;
        return workspace.account_id === activeAccountId;
      }) ??
      null;

    if (match) {
      setWorkspaceScope(match.scope);
      setSelectedWorkspaceId(match.id);
      if (match.run_id) setActiveRunId(match.run_id);
      return;
    }

    setSelectedWorkspaceId(null);
  }

  async function startRun(nextMode: RunMode = mode, background = false, realConfirmed = false) {
    if (nextMode === "real" && !realConfirmed) {
      setPendingRealAction({
        type: "start_run",
        payload: { benchmark: selectedBenchmark, mode: "real", claim_level: "workflow_proxy", background }
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
        target_solution_count: runConfig.target_solution_count,
        max_iterations: runBudgetPreview.max_iterations,
        parallel_mutations: runConfig.parallel_mutations,
        selector_vote_count: runConfig.selector_vote_count,
        max_children_per_node: runConfig.max_children_per_node,
        agent_models: activeAgentModels(),
        selected_algorithm_ids: selectedAlgorithmIds,
        manual_strategy_locks: activeStrategyLocks(strategyLocks),
        branch_context: strategyLockBranchContext(strategyLocks),
        problem_intake: problemPlan?.recommended_benchmark.name === selectedBenchmark ? problemPlan.problem_intake : {},
        planner_snapshot: problemPlan?.recommended_benchmark.name === selectedBenchmark ? problemPlan.planner_snapshot : {},
        claim_level: "workflow_proxy",
        background,
        real_confirmed: realConfirmed
      });
      setActiveRun(run);
      setActiveRunId(run.run_id);
      await refreshRuns();
      await refreshRunEvidence(run.run_id);
      if (background) {
        pollBackgroundRun(run.run_id, activeAccountId);
      }
      addAssistantMessage(`${nextMode} run 已登记：${run.run_id}`);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function previewRunReadiness(realConfirmed = false) {
    setBusy(true);
    setError(null);
    try {
      const report = await api.previewReadiness({
        benchmark: selectedBenchmark,
        mode,
        account_id: activeAccountId,
        target_solution_count: runConfig.target_solution_count,
        max_iterations: runBudgetPreview.max_iterations,
        parallel_mutations: runConfig.parallel_mutations,
        selector_vote_count: runConfig.selector_vote_count,
        max_children_per_node: runConfig.max_children_per_node,
        selected_algorithm_ids: selectedAlgorithmIds,
        manual_strategy_locks: activeStrategyLocks(strategyLocks),
        branch_context: strategyLockBranchContext(strategyLocks),
        problem_intake: problemPlan?.recommended_benchmark.name === selectedBenchmark ? problemPlan.problem_intake : {},
        planner_snapshot: problemPlan?.recommended_benchmark.name === selectedBenchmark ? problemPlan.planner_snapshot : {},
        claim_level: "workflow_proxy",
        real_confirmed: realConfirmed
      });
      setReadinessReport(report);
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
        account_id: activeAccountId,
        target_solution_count: runConfig.target_solution_count,
        parallel_mutations: runConfig.parallel_mutations,
        selector_vote_count: runConfig.selector_vote_count,
        max_children_per_node: runConfig.max_children_per_node,
        selected_algorithm_ids: selectedAlgorithmIds,
        agent_models: activeAgentModels(),
        claim_level: "workflow_proxy"
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
        await startRunFromAction(action, false);
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
        await selectWorkspaceFromCodeServerAction(action);
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

  async function confirmRealAction() {
    const action = pendingRealAction;
    if (!action) return;
    setPendingRealAction(null);
    if (action.type === "start_run") {
      await startRunFromAction(action, true);
    }
    if (action.type === "resume_run" && action.run_id) {
      setBusy(true);
      setError(null);
      try {
        const run = await api.resumeRun(action.run_id, selectedBenchmark, "real", activeAccountId, true);
        setActiveRun(run);
        setActiveRunId(run.run_id);
        await refreshRuns();
      } catch (exc) {
        setError(String(exc));
      } finally {
        setBusy(false);
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

  async function startRunFromAction(action: SolverAction, realConfirmed: boolean) {
    const payload = action.payload ?? {};
    const actionMode = asRunMode(payload.mode, mode);
    const nextBenchmark = typeof payload.benchmark === "string" ? payload.benchmark : selectedBenchmark;
    const nextRunConfig: RunConfig = {
      target_solution_count: payloadNumber(payload, "target_solution_count", runConfig.target_solution_count),
      max_iterations: payloadNumber(payload, "max_iterations", runConfig.max_iterations),
      parallel_mutations: payloadNumber(payload, "parallel_mutations", runConfig.parallel_mutations),
      selector_vote_count: payloadNumber(payload, "selector_vote_count", runConfig.selector_vote_count),
      max_children_per_node: payloadNumber(payload, "max_children_per_node", runConfig.max_children_per_node)
    };
    const nextAlgorithmIds = payloadStringArray(payload, "selected_algorithm_ids", selectedAlgorithmIds);
    const nextAgentModels = payloadAgentModels(payload.agent_models, activeAgentModels());
    const nextProblemIntake = payloadRecord(payload.problem_intake, {});
    const nextPlannerSnapshot = payloadRecord(payload.planner_snapshot, {});
    const nextClaimLevel = payload.claim_level === "paper_workflow" ? "paper_workflow" : "workflow_proxy";

    setBusy(true);
    setError(null);
    try {
      const run = await api.startRun({
        benchmark: nextBenchmark,
        mode: actionMode,
        account_id: activeAccountId,
        target_solution_count: nextRunConfig.target_solution_count,
        max_iterations: nextRunConfig.max_iterations,
        parallel_mutations: nextRunConfig.parallel_mutations,
        selector_vote_count: nextRunConfig.selector_vote_count,
        max_children_per_node: nextRunConfig.max_children_per_node,
        agent_models: nextAgentModels,
        selected_algorithm_ids: nextAlgorithmIds,
        manual_strategy_locks: activeStrategyLocks(strategyLocks),
        branch_context: strategyLockBranchContext(strategyLocks),
        problem_intake: nextProblemIntake,
        planner_snapshot: nextPlannerSnapshot,
        claim_level: nextClaimLevel,
        background: Boolean(payload.background),
        real_confirmed: realConfirmed
      });
      setSelectedBenchmark(nextBenchmark);
      setRunConfig(nextRunConfig);
      setSelectedAlgorithmIds(nextAlgorithmIds);
      setProblemPlan(null);
      setActiveRun(run);
      setActiveRunId(run.run_id);
      await refreshRuns();
      await refreshRunEvidence(run.run_id);
      if (payload.background) {
        pollBackgroundRun(run.run_id, activeAccountId);
      }
      addAssistantMessage(`${actionMode} run 已登记：${run.run_id}`);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
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

  function selectPaperTask(task: PaperTask) {
    setSelectedPaperSection(task.paper_section);
    const preferred =
      task.benchmarks.find((benchmark) => benchmark.fidelity_level === "faithful-small") ??
      task.benchmarks[0];
    if (preferred) {
      setSelectedBenchmark(preferred.name);
      setReadinessReport(null);
    }
  }

  function updateRunConfig(next: Partial<RunConfig>) {
    setReadinessReport(null);
    setRunConfig((current) => {
      const merged = { ...current, ...next };
      const parallelMutations = normalizeInteger(merged.parallel_mutations, current.parallel_mutations, 1);
      const targetSolutionCount = normalizeInteger(merged.target_solution_count, current.target_solution_count, 1);
      const maxIterations = Math.max(
        0,
        Math.ceil(Math.max(0, targetSolutionCount - 1) / parallelMutations)
      );
      return {
        target_solution_count: targetSolutionCount,
        max_iterations: maxIterations,
        parallel_mutations: parallelMutations,
        selector_vote_count: normalizeInteger(merged.selector_vote_count, current.selector_vote_count, 1),
        max_children_per_node: normalizeInteger(merged.max_children_per_node, current.max_children_per_node, 1)
      };
    });
  }

  function updateAgentModel(role: string, next: Partial<AgentModelConfig>) {
    setAgentModels((current) => {
      const defaults = defaultAgentModelConfig(role, agentRoles);
      const existing = current[role] ?? defaults;
      const merged = { ...existing, ...next };
      if (!merged.model.trim()) {
        const rest = { ...current };
        delete rest[role];
        return rest;
      }
      return {
        ...current,
        [role]: {
          model: merged.model,
          temperature: Number.isFinite(merged.temperature) ? merged.temperature : existing.temperature,
          reasoning_effort: merged.reasoning_effort
        }
      };
    });
  }

  function activeAgentModels() {
    return Object.fromEntries(
      Object.entries(agentModels)
        .filter(([, config]) => config.model.trim())
        .map(([role, config]) => [
          role,
          {
            model: config.model.trim(),
            temperature: config.temperature,
            reasoning_effort: config.reasoning_effort
          }
        ])
    );
  }

  function toggleAlgorithm(algorithmId: string) {
    setReadinessReport(null);
    setSelectedAlgorithmIds((current) =>
      current.includes(algorithmId)
        ? current.filter((item) => item !== algorithmId)
        : [...current, algorithmId]
    );
  }

  function addStrategyLock() {
    setReadinessReport(null);
    setStrategyLocks((current) => [
      ...current,
      {
        lock_id: nextStrategyLockId(current),
        kind: "constraint",
        text: "",
        scope: "all_branches",
        required: true
      }
    ]);
  }

  function updateStrategyLock(lockId: string, next: Partial<StrategyLock>) {
    setReadinessReport(null);
    setStrategyLocks((current) =>
      current.map((lock) => (lock.lock_id === lockId ? { ...lock, ...next } : lock))
    );
  }

  function removeStrategyLock(lockId: string) {
    setReadinessReport(null);
    setStrategyLocks((current) => current.filter((lock) => lock.lock_id !== lockId));
  }

  function updateProblemIntake(next: Partial<ProblemIntakeState>) {
    setReadinessReport(null);
    setProblemIntake((current) => ({ ...current, ...next }));
  }

  async function planProblemRun() {
    setBusy(true);
    setError(null);
    try {
      const plan = await api.planProblem({
        ...problemIntake,
        mode,
        target_solution_count: runConfig.target_solution_count,
        parallel_mutations: runConfig.parallel_mutations,
        selector_vote_count: runConfig.selector_vote_count,
        max_children_per_node: runConfig.max_children_per_node,
        selected_algorithm_ids: selectedAlgorithmIds,
        agent_models: activeAgentModels(),
        claim_level: "workflow_proxy"
      });
      setProblemPlan(plan);
      setReadinessReport(null);
      setSelectedBenchmark(plan.recommended_benchmark.name);
      setSelectedAlgorithmIds(plan.selected_algorithm_ids);
      setRunConfig((current) => ({
        ...current,
        target_solution_count: plan.run_config.target_solution_count,
        max_iterations: plan.run_config.max_iterations,
        parallel_mutations: plan.run_config.parallel_mutations,
        selector_vote_count: plan.run_config.selector_vote_count,
        max_children_per_node: plan.run_config.max_children_per_node
      }));
      setMode(plan.run_config.mode);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
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

  function setBoundedAgentPanelWidth(nextWidth: number) {
    setAgentPanelWidth(Math.min(AGENT_PANEL_MAX_WIDTH, Math.max(AGENT_PANEL_MIN_WIDTH, Math.round(nextWidth))));
  }

  function startAgentPanelResize(event: ReactPointerEvent<HTMLButtonElement>) {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const startX = event.clientX;
    const startWidth = agentPanelWidth;
    const resizeHandle = event.currentTarget;

    const handlePointerMove = (moveEvent: PointerEvent) => {
      setBoundedAgentPanelWidth(startWidth + startX - moveEvent.clientX);
    };
    const stopResize = () => {
      if (resizeHandle.hasPointerCapture(event.pointerId)) {
        resizeHandle.releasePointerCapture(event.pointerId);
      }
      document.body.classList.remove("resizing-agent-panel");
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopResize);
    };

    document.body.classList.add("resizing-agent-panel");
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopResize);
  }

  function resizeAgentPanelBy(delta: number) {
    setBoundedAgentPanelWidth(agentPanelWidth + delta);
  }

  const idePageStyle = selectedWorkspace
    ? ({ "--agent-panel-width": `${agentPanelWidth}px` } as CSSProperties)
    : undefined;

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
        <section className="chatgpt-page" aria-label="ChatUI 页面" data-testid="page-chat">
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
        <section
          className={selectedWorkspace ? "ide-page workspace-open" : "ide-page workspace-select"}
          aria-label="AI IDE 页面"
          data-testid="page-ide"
          style={idePageStyle}
        >
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
                onConfirmRealAction={confirmRealAction}
                onSend={sendMessage}
                onToggle={() => setAgentCollapsed((current) => !current)}
                onDismissRealAction={() => setPendingRealAction(null)}
                onResizeReset={() => setAgentPanelWidth(AGENT_PANEL_DEFAULT_WIDTH)}
                onResizeStart={startAgentPanelResize}
                onResizeStep={resizeAgentPanelBy}
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
        <section className="library-page" aria-label="算法库页面" data-testid="page-library">
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
            agentModels={agentModels}
            agentRoles={agentRoles}
            algorithms={algorithms}
            artifactPayload={artifactPayload}
            events={filteredEvents}
            filter={traceFilter}
            language={libraryLanguage}
            mode={mode}
            paperTasks={paperTasks}
            problemIntake={problemIntake}
            problemPlan={problemPlan}
            readinessReport={readinessReport}
            runBudgetPreview={runBudgetPreview}
            runConfig={runConfig}
            runs={runs}
            selected={selected}
            selectedAlgorithmIds={selectedAlgorithmIds}
            selectedArtifactPath={selectedArtifactPath}
            selectedPaperTask={selectedPaperTask}
            selectorVotes={selectorVotes}
            solutionsPayload={solutionsPayload}
            strategyLocks={strategyLocks}
            onAddStrategyLock={addStrategyLock}
            onFilterChange={setTraceFilter}
            onLanguageChange={setLibraryLanguage}
            onModeChange={setMode}
            onOpenArtifacts={() => setSelectedArtifactPath("trace_summary.json")}
            onOpenCode={() => setActivePage("ide")}
            onPlanProblemRun={planProblemRun}
            onPreviewReadiness={() => previewRunReadiness(false)}
            onRefreshRuns={refreshRuns}
            onRemoveStrategyLock={removeStrategyLock}
            onRunConfigChange={updateRunConfig}
            onSelectArtifact={setSelectedArtifactPath}
            onSelectPaperTask={selectPaperTask}
            onSelectRun={selectRun}
            onStartConfiguredRun={() => startRun(mode, false)}
            onStartMock={() => startRun("mock", false)}
            onToggleAlgorithm={toggleAlgorithm}
            onUpdateStrategyLock={updateStrategyLock}
            onUpdateProblemIntake={updateProblemIntake}
            onUpdateAgentModel={updateAgentModel}
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
            data-testid={`nav-${page.key}`}
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
          data-testid="chat-main-input"
          value={mainPrompt}
          onChange={(event) => onChange(event.target.value)}
          placeholder="给 AgenticSciML 发消息"
        />
        <div className="composer-tools">
          <AssistantModeSwitch mode={assistantMode} settings={modeSettings} onChange={onAssistantModeChange} />
          <button type="button" title="Artifact context">
            <Database size={15} />
          </button>
          <button className="send-button" data-testid="chat-main-send" disabled={busy} type="submit" title="发送">
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
            data-testid={`assistant-mode-${item}`}
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
  agentModels,
  agentRoles,
  algorithms,
  artifactPayload,
  events,
  filter,
  language,
  mode,
  paperTasks,
  problemIntake,
  problemPlan,
  readinessReport,
  runBudgetPreview,
  runConfig,
  runs,
  selected,
  selectedAlgorithmIds,
  selectedArtifactPath,
  selectedPaperTask,
  selectorVotes,
  solutionsPayload,
  strategyLocks,
  onAddStrategyLock,
  onFilterChange,
  onLanguageChange,
  onModeChange,
  onOpenArtifacts,
  onOpenCode,
  onPlanProblemRun,
  onPreviewReadiness,
  onRefreshRuns,
  onRemoveStrategyLock,
  onRunConfigChange,
  onSelectArtifact,
  onSelectPaperTask,
  onSelectRun,
  onStartConfiguredRun,
  onStartMock,
  onToggleAlgorithm,
  onUpdateStrategyLock,
  onUpdateProblemIntake,
  onUpdateAgentModel
}: {
  activeRun: RunSummary | null;
  activeRunId: string | null;
  agentModels: Record<string, AgentModelConfig>;
  agentRoles: AgentRole[];
  algorithms: AlgorithmSpec[];
  artifactPayload: ArtifactPayload | null;
  events: string[];
  filter: string;
  language: LibraryLanguage;
  mode: RunMode;
  paperTasks: PaperTask[];
  problemIntake: ProblemIntakeState;
  problemPlan: ProblemRunPlan | null;
  readinessReport: ReadinessReport | null;
  runBudgetPreview: { max_iterations: number; planned_solution_budget: number };
  runConfig: RunConfig;
  runs: RunSummary[];
  selected?: Benchmark;
  selectedAlgorithmIds: string[];
  selectedArtifactPath: string;
  selectedPaperTask: PaperTask | null;
  selectorVotes: SelectorVotesPayload | null;
  solutionsPayload: SolutionsPayload | null;
  strategyLocks: StrategyLock[];
  onAddStrategyLock: () => void;
  onFilterChange: (value: string) => void;
  onLanguageChange: (language: LibraryLanguage) => void;
  onModeChange: (mode: RunMode) => void;
  onOpenArtifacts: () => void;
  onOpenCode: () => void;
  onPlanProblemRun: () => void;
  onPreviewReadiness: () => void;
  onRefreshRuns: () => void;
  onRemoveStrategyLock: (lockId: string) => void;
  onRunConfigChange: (next: Partial<RunConfig>) => void;
  onSelectArtifact: (value: string) => void;
  onSelectPaperTask: (task: PaperTask) => void;
  onSelectRun: (run: RunSummary) => void;
  onStartConfiguredRun: () => void;
  onStartMock: () => void;
  onToggleAlgorithm: (algorithmId: string) => void;
  onUpdateStrategyLock: (lockId: string, next: Partial<StrategyLock>) => void;
  onUpdateProblemIntake: (next: Partial<ProblemIntakeState>) => void;
  onUpdateAgentModel: (role: string, next: Partial<AgentModelConfig>) => void;
}) {
  const paperAlgorithmIds = new Set((selectedPaperTask?.algorithms ?? []).map((algorithm) => algorithm.id));
  const scopedAlgorithms = [...algorithms].sort((left, right) => {
    const leftPinned = paperAlgorithmIds.has(left.id) ? 0 : 1;
    const rightPinned = paperAlgorithmIds.has(right.id) ? 0 : 1;
    return leftPinned - rightPinned || left.name.localeCompare(right.name);
  });
  return (
    <section className="library-content paper-lab">
      <section className="paper-lab-head">
        <div>
          <p className="eyebrow">Paper Workflow Evidence</p>
          <h2>算法库 / 论文工作流证据</h2>
          <span>S1 任务、faithful-small / proxy benchmark、selector votes、solution loss 和本地 artifact 只读证据，不表示论文成绩复现。</span>
        </div>
        <StatusBadge tone="info">not paper-score evidence</StatusBadge>
      </section>
      <PaperTaskTabs tasks={paperTasks} selected={selectedPaperTask} onSelect={onSelectPaperTask} />
      <div className="paper-lab-grid">
        <ProblemIntakePanel
          problemIntake={problemIntake}
          problemPlan={problemPlan}
          onChange={onUpdateProblemIntake}
          onPlan={onPlanProblemRun}
        />
        <RunConfigPanel
          budgetPreview={runBudgetPreview}
          busy={false}
          mode={mode}
          readinessReport={readinessReport}
          runConfig={runConfig}
          selectedBenchmark={selected?.name ?? "unknown"}
          onModeChange={onModeChange}
          onPreviewReadiness={onPreviewReadiness}
          onRunConfigChange={onRunConfigChange}
          onStartRun={onStartConfiguredRun}
        />
        <StrategyLocksPanel
          locks={strategyLocks}
          onAdd={onAddStrategyLock}
          onRemove={onRemoveStrategyLock}
          onUpdate={onUpdateStrategyLock}
        />
      </div>
      <PaperTaskDetail task={selectedPaperTask} selectedBenchmark={selected?.name ?? null} />
      <RoleModelPanel agentModels={agentModels} roles={agentRoles} onUpdate={onUpdateAgentModel} />
      <EvidencePanel selectorVotes={selectorVotes} solutionsPayload={solutionsPayload} />
      <DataRegion title="Paper primitive catalog">
        <div className="catalog-toolbar" aria-label="算法库语言切换">
          <div>
            <span>{language === "zh" ? "算法说明" : "Algorithm notes"}</span>
            <strong>{language === "zh" ? "特点 / 对应问题" : "Features / Problem fit"}</strong>
          </div>
          <div className="language-toggle" role="group" aria-label="Algorithm catalog language">
            <button
              className={language === "zh" ? "selected" : ""}
              data-testid="algorithm-language-zh"
              type="button"
              onClick={() => onLanguageChange("zh")}
            >
              中文
            </button>
            <button
              className={language === "en" ? "selected" : ""}
              data-testid="algorithm-language-en"
              type="button"
              onClick={() => onLanguageChange("en")}
            >
              EN
            </button>
          </div>
        </div>
        <div className="algorithm-context">
          {selectedPaperTask ? (
            <p>
              {selectedPaperTask.paper_section} 当前绑定 {selectedPaperTask.algorithms.length} 个 paper reference primitive；
              score、champion 和科学声明仍只来自 evaluator 与 run artifacts。
            </p>
          ) : (
            <p>尚未加载 S1 task mapping。</p>
          )}
        </div>
        <AlgorithmCatalog
          algorithms={scopedAlgorithms}
          embedded
          language={language}
          selectedIds={selectedAlgorithmIds}
          onToggle={onToggleAlgorithm}
        />
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

function PaperTaskTabs({
  onSelect,
  selected,
  tasks
}: {
  onSelect: (task: PaperTask) => void;
  selected: PaperTask | null;
  tasks: PaperTask[];
}) {
  return (
    <div className="paper-task-tabs" aria-label="Paper S1 tasks">
      {tasks.map((task) => (
        <button
          className={selected?.paper_section === task.paper_section ? "selected" : ""}
          key={task.paper_section}
          type="button"
          onClick={() => onSelect(task)}
        >
          <span>{task.paper_section}</span>
          <strong>{task.title}</strong>
        </button>
      ))}
      {tasks.length === 0 ? <span className="muted">S1 task mapping 尚未加载。</span> : null}
    </div>
  );
}

function ProblemIntakePanel({
  onChange,
  onPlan,
  problemIntake,
  problemPlan
}: {
  onChange: (next: Partial<ProblemIntakeState>) => void;
  onPlan: () => void;
  problemIntake: ProblemIntakeState;
  problemPlan: ProblemRunPlan | null;
}) {
  const canPlan = problemIntake.problem_statement.trim().length >= 20;
  return (
    <DataRegion title="Problem Intake">
      <div className="problem-intake">
        <label>
          <span>完整问题描述</span>
          <textarea
            value={problemIntake.problem_statement}
            onChange={(event) => onChange({ problem_statement: event.target.value })}
            placeholder="描述科学问题、输入输出、约束、可用数据和期望评价方式"
          />
        </label>
        <div className="problem-intake-grid">
          <label>
            <span>Requirements</span>
            <textarea
              value={problemIntake.requirements}
              onChange={(event) => onChange({ requirements: event.target.value })}
              placeholder="运行约束、禁止事项、依赖、预算"
            />
          </label>
          <label>
            <span>Evaluation</span>
            <textarea
              value={problemIntake.evaluation_criteria}
              onChange={(event) => onChange({ evaluation_criteria: event.target.value })}
              placeholder="metric、loss、验证方式"
            />
          </label>
          <label>
            <span>Data</span>
            <textarea
              value={problemIntake.data_description}
              onChange={(event) => onChange({ data_description: event.target.value })}
              placeholder="数据形状、变量、train/validation 边界"
            />
          </label>
        </div>
        <button className="icon-text-button full-width" disabled={!canPlan} type="button" onClick={onPlan}>
          <Sparkles size={15} />
          自动评选 benchmark 与解法
        </button>
        {problemPlan ? <ProblemPlanSummary plan={problemPlan} /> : null}
      </div>
    </DataRegion>
  );
}

function ProblemPlanSummary({ plan }: { plan: ProblemRunPlan }) {
  return (
    <div className="problem-plan-summary">
      <div>
        <span>recommended benchmark</span>
        <strong>{plan.recommended_benchmark.name}</strong>
      </div>
      <div>
        <span>selected algorithms</span>
        <strong>{plan.selected_algorithm_ids.join(", ") || "none"}</strong>
      </div>
      <p>{plan.claim_boundary}</p>
      {plan.custom_problem_package ? (
        <div className="claim-gate-mini">
          <span>{plan.custom_problem_package.status}</span>
          <strong>{plan.custom_problem_package.evaluator_trust_level}</strong>
          <small>
            paper_equivalent={String(plan.custom_problem_package.paper_benchmark_equivalent)} · replacement_required=
            {String(plan.custom_problem_package.requires_replacement_for_scientific_claim)}
          </small>
        </div>
      ) : null}
      {plan.warnings.map((warning) => (
        <small key={warning}>{warning}</small>
      ))}
    </div>
  );
}

function StrategyLocksPanel({
  locks,
  onAdd,
  onRemove,
  onUpdate
}: {
  locks: StrategyLock[];
  onAdd: () => void;
  onRemove: (lockId: string) => void;
  onUpdate: (lockId: string, next: Partial<StrategyLock>) => void;
}) {
  return (
    <DataRegion title="Strategy Locks">
      <div className="strategy-locks">
        {locks.map((lock) => (
          <div className="strategy-lock-row" key={lock.lock_id}>
            <div className="strategy-lock-header">
              <select
                value={lock.kind}
                onChange={(event) => onUpdate(lock.lock_id, { kind: event.target.value as StrategyLock["kind"] })}
              >
                <option value="constraint">constraint</option>
                <option value="invariant">invariant</option>
                <option value="mathematical_intuition">mathematical intuition</option>
                <option value="modeling_choice">modeling choice</option>
                <option value="assumption">assumption</option>
              </select>
              <select
                value={lock.scope}
                onChange={(event) => onUpdate(lock.lock_id, { scope: event.target.value as StrategyLock["scope"] })}
              >
                <option value="all_branches">all branches</option>
                <option value="selected_algorithms">selected algorithms</option>
              </select>
              <button className="icon-button" type="button" onClick={() => onRemove(lock.lock_id)} title="移除 strategy lock">
                <X size={15} />
              </button>
            </div>
            <textarea
              value={lock.text}
              onChange={(event) => onUpdate(lock.lock_id, { text: event.target.value })}
              placeholder="例如：保持 bias-free linear branch net，所有 sibling branch 都必须继承"
            />
          </div>
        ))}
        {locks.length === 0 ? <p className="muted">尚未设置人工策略锁。</p> : null}
        <button className="icon-text-button full-width secondary" type="button" onClick={onAdd}>
          <Plus size={15} />
          添加 strategy lock
        </button>
      </div>
    </DataRegion>
  );
}

function PaperTaskDetail({
  selectedBenchmark,
  task
}: {
  selectedBenchmark: string | null;
  task: PaperTask | null;
}) {
  if (!task) {
    return (
      <DataRegion title="Paper Tasks">
        <p className="muted">等待 /api/paper-tasks。</p>
      </DataRegion>
    );
  }
  return (
    <DataRegion title={`${task.paper_section} / ${task.title}`}>
      <div className="paper-task-detail">
        <p>{task.summary}</p>
        <div className="paper-task-columns">
          <div>
            <strong>Benchmarks</strong>
            {task.benchmarks.map((benchmark) => (
              <div
                className={benchmark.name === selectedBenchmark ? "paper-benchmark-row selected" : "paper-benchmark-row"}
                key={benchmark.name}
              >
                <span>{benchmark.name}</span>
                <small>{benchmark.fidelity_level} · {benchmark.metric}</small>
              </div>
            ))}
          </div>
          <div>
            <strong>Reference primitives</strong>
            {task.reference_primitives.map((primitive) => (
              <span className="primitive-chip" key={primitive}>{primitive}</span>
            ))}
          </div>
        </div>
        <div className="claim-boundary">
          <AlertTriangle size={15} />
          <span>{task.claim_boundary}</span>
        </div>
      </div>
    </DataRegion>
  );
}

function RunConfigPanel({
  budgetPreview,
  busy,
  mode,
  readinessReport,
  runConfig,
  selectedBenchmark,
  onModeChange,
  onPreviewReadiness,
  onRunConfigChange,
  onStartRun
}: {
  budgetPreview: { max_iterations: number; planned_solution_budget: number };
  busy: boolean;
  mode: RunMode;
  readinessReport: ReadinessReport | null;
  runConfig: RunConfig;
  selectedBenchmark: string;
  onModeChange: (mode: RunMode) => void;
  onPreviewReadiness: () => void;
  onRunConfigChange: (next: Partial<RunConfig>) => void;
  onStartRun: () => void;
}) {
  return (
    <DataRegion title="Run Config">
      <div className="run-config-panel">
        <div className="run-config-mode">
          <span>mode</span>
          <div className="segmented compact" aria-label="Paper run mode">
            {(["mock", "dry_run", "real"] as const).map((item) => (
              <button
                className={mode === item ? "selected" : ""}
                key={item}
                type="button"
                onClick={() => onModeChange(item)}
              >
                {item}
              </button>
            ))}
          </div>
        </div>
        <div className="config-grid">
          <NumberField
            label="target solutions"
            min={1}
            value={runConfig.target_solution_count}
            onChange={(value) => onRunConfigChange({ target_solution_count: value })}
          />
          <NumberField
            label="max iterations"
            min={0}
            value={budgetPreview.max_iterations}
            onChange={(value) => onRunConfigChange({ max_iterations: value, target_solution_count: 1 + value * runConfig.parallel_mutations })}
          />
          <NumberField
            label="parallel mutations"
            min={1}
            value={runConfig.parallel_mutations}
            onChange={(value) => onRunConfigChange({ parallel_mutations: value })}
          />
          <NumberField
            label="selector votes"
            min={1}
            value={runConfig.selector_vote_count}
            onChange={(value) => onRunConfigChange({ selector_vote_count: value })}
          />
          <NumberField
            label="max children/node"
            min={1}
            value={runConfig.max_children_per_node}
            onChange={(value) => onRunConfigChange({ max_children_per_node: value })}
          />
        </div>
        <div className="budget-preview">
          <div>
            <span>actual budget</span>
            <strong>{budgetPreview.planned_solution_budget}</strong>
          </div>
          <div>
            <span>root + children</span>
            <strong>1 + {budgetPreview.max_iterations * runConfig.parallel_mutations}</strong>
          </div>
          <div>
            <span>benchmark</span>
            <strong>{selectedBenchmark}</strong>
          </div>
          <div>
            <span>operator scheduler</span>
            <strong>auto-audited</strong>
          </div>
        </div>
        <button className="icon-text-button full-width" type="button" onClick={onStartRun}>
          <Play size={15} />
          启动配置 run
        </button>
        <button className="icon-text-button full-width secondary" disabled={busy} type="button" onClick={onPreviewReadiness}>
          <AlertTriangle size={15} />
          检查 run readiness
        </button>
        {readinessReport ? <RunReadinessSummary report={readinessReport} /> : null}
        {mode === "real" ? (
          <p className="warning-text">real mode 仍需要后端显式开关与二次确认，不会绕过预算或 claim boundary。</p>
        ) : null}
      </div>
    </DataRegion>
  );
}

function RunReadinessSummary({ report }: { report: ReadinessReport }) {
  const visibleChecks = report.checks
    .filter((check) => check.severity !== "info" || !check.passed)
    .slice(0, 5);
  return (
    <div className={`readiness-summary ${report.status}`}>
      <div className="readiness-title">
        <strong>{report.status}</strong>
        <span>
          {report.summary.blocker_count} blockers / {report.summary.warning_count} warnings
        </span>
      </div>
      <div className="readiness-meta">
        <span>{report.benchmark_fidelity_preview[0]?.fidelity_level ?? "unknown"} fidelity</span>
        <span>{report.algorithm_seed_preview.length} strategy seeds</span>
        <span>{report.claim_gate.claim_level} claim</span>
        <span>KB {report.kb_manifest?.coverage_status ?? "unknown"}</span>
      </div>
      <div className="claim-gate-mini">
        <span>{report.claim_gate.status}</span>
        <strong>{report.claim_gate.evaluator_trust_level}</strong>
        <small>
          paper={String(report.claim_gate.paper_level_claim_supported)} · scientific=
          {String(report.claim_gate.scientific_claim_supported)} · selector=
          {String(report.selector_panel_preview?.configured_heterogeneous ?? false)}
        </small>
      </div>
      {visibleChecks.map((check) => (
        <div className={`readiness-check ${check.severity}`} key={check.check_id}>
          <span>{check.severity}</span>
          <p>{check.message}</p>
        </div>
      ))}
      <small>{report.claim_boundary}</small>
    </div>
  );
}

function NumberField({
  label,
  min,
  onChange,
  value
}: {
  label: string;
  min: number;
  onChange: (value: number) => void;
  value: number;
}) {
  return (
    <label className="number-field">
      <span>{label}</span>
      <input
        min={min}
        type="number"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function RoleModelPanel({
  agentModels,
  onUpdate,
  roles
}: {
  agentModels: Record<string, AgentModelConfig>;
  onUpdate: (role: string, next: Partial<AgentModelConfig>) => void;
  roles: AgentRole[];
}) {
  return (
    <DataRegion title="Layered model routing">
      <div className="role-model-panel">
        <p>
          默认使用后端单一 adapter；只有填写 role override 时才按层路由。reasoning_effort 会写入 audit，
          并传给支持该参数的 provider。
        </p>
        <div className="role-model-table">
          <table>
            <thead>
              <tr>
                <th>role</th>
                <th>layer</th>
                <th>model override</th>
                <th>temp</th>
                <th>thinking</th>
              </tr>
            </thead>
            <tbody>
              {roles.map((role) => {
                const config = agentModels[role.role] ?? defaultAgentModelConfig(role.role, roles);
                return (
                  <tr key={role.role}>
                    <td>{role.label}</td>
                    <td>{role.kind}</td>
                    <td>
                      <input
                        value={config.model}
                        onChange={(event) => onUpdate(role.role, { model: event.target.value })}
                        placeholder="default"
                      />
                    </td>
                    <td>
                      <input
                        min={0}
                        max={2}
                        step={0.05}
                        type="number"
                        value={config.temperature}
                        onChange={(event) => onUpdate(role.role, { temperature: Number(event.target.value) })}
                      />
                    </td>
                    <td>
                      <select
                        value={config.reasoning_effort ?? "medium"}
                        onChange={(event) =>
                          onUpdate(role.role, { reasoning_effort: event.target.value as ReasoningEffort })
                        }
                      >
                        <option value="low">low</option>
                        <option value="medium">medium</option>
                        <option value="high">high</option>
                        <option value="xhigh">xhigh</option>
                      </select>
                    </td>
                  </tr>
                );
              })}
              {roles.length === 0 ? (
                <tr>
                  <td colSpan={5}>等待 /api/agent-roles。</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
    </DataRegion>
  );
}

function EvidencePanel({
  selectorVotes,
  solutionsPayload
}: {
  selectorVotes: SelectorVotesPayload | null;
  solutionsPayload: SolutionsPayload | null;
}) {
  return (
    <div className="evidence-grid">
      <DataRegion title="Selector votes">
        <SelectorVotesView payload={selectorVotes} />
      </DataRegion>
      <DataRegion title="Solution loss / tree">
        <SolutionsTable payload={solutionsPayload} />
      </DataRegion>
      <DataRegion title="Local figures">
        <FigureArtifactsView figures={solutionsPayload?.figures ?? []} />
      </DataRegion>
    </div>
  );
}

function SelectorVotesView({ payload }: { payload: SelectorVotesPayload | null }) {
  if (!payload) return <p className="muted">选择 run 后显示 selector votes。</p>;
  if (!payload.available) return <p className="muted">该 run 没有 reports/selector_votes.json。</p>;
  const counts = Object.entries(payload.vote_counts);
  return (
    <div className="votes-view">
      <div className="vote-counts">
        {counts.map(([nodeId, count]) => (
          <span key={nodeId}>
            <strong>{nodeId}</strong>
            {count}
          </span>
        ))}
      </div>
      <div className="vote-list">
        {payload.votes.slice(0, 6).map((vote, index) => (
          <code key={`${index}-${String(vote.candidate_id ?? vote.node_id ?? "vote")}`}>
            {String(vote.candidate_id ?? vote.node_id ?? "vote")} · {String(vote.rationale ?? vote.reason ?? "no rationale")}
          </code>
        ))}
      </div>
    </div>
  );
}

function SolutionsTable({ payload }: { payload: SolutionsPayload | null }) {
  const rows = payload?.solutions ?? [];
  const health = payload?.evolution_health;
  const innovation = payload?.innovation_report;
  const operatorCount =
    innovation?.operator_count ??
    (health?.operator_health ? Object.keys(health.operator_health).length : undefined);
  return (
    <div className="solution-table-stack">
      <div className="compact-metrics">
        <span>operator scheduler {health?.operator_scheduler_mode ?? "auto-audited"}</span>
        <span>operators {operatorCount ?? "n/a"}</span>
        <span>
          assignments {health?.operator_assignment_count ?? "n/a"}/
          {health?.operator_assignment_expected_count ?? "n/a"}
        </span>
        <span>missing assignments {(health?.missing_operator_assignment_nodes ?? []).length}</span>
        <span>unique code {health?.unique_code_count ?? "n/a"}</span>
        <span>duplicates {health?.duplicate_code_count ?? "n/a"}</span>
        <span>plateau {health?.max_plateau_length ?? "n/a"}</span>
        <span>best improvement {formatScore(health?.best_improvement)}</span>
        <span>innovation axes {innovation?.novelty_axis_count ?? "n/a"}</span>
        <span>candidate emergence {innovation?.candidate_emergent_count ?? "n/a"}</span>
      </div>
      {(health?.warnings ?? []).length ? (
        <div className="inline-warnings">
          {(health?.warnings ?? []).map((warning) => (
            <span key={warning}>{warning}</span>
          ))}
        </div>
      ) : null}
      {innovation?.available ? (
        <div className="inline-warnings muted">
          <span>{innovation.innovation_claim_level ?? "workflow_exploration_only"}</span>
          <span>scientific novelty {String(innovation.scientific_novelty_supported ?? false)}</span>
          {(innovation.top_axes ?? []).filter(Boolean).map((axis) => (
            <span key={axis ?? "axis"}>{axis}</span>
          ))}
        </div>
      ) : null}
      <div className="table-wrap solution-table">
        <table>
          <thead>
            <tr>
              <th>node</th>
              <th>parent</th>
              <th>status</th>
              <th>metric</th>
              <th>loss/score</th>
              <th>delta</th>
              <th>operator</th>
              <th>kb</th>
              <th>mutation</th>
              <th>emergence</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 12).map((solution) => (
              <tr key={solution.node_id}>
                <td>{solution.node_id}</td>
                <td>{solution.parent_id ?? "root"}</td>
                <td>{solution.status ?? "unknown"}</td>
                <td>{solution.metric ?? "metric"}</td>
                <td>{formatScore(solution.loss ?? solution.score)}</td>
                <td>{formatScore(solution.score_delta_from_parent)}</td>
                <td>{formatOperatorAssignment(solution.operator_assignment)}</td>
                <td>{formatKbApplication(solution.kb_application)}</td>
                <td>{formatMutationEffect(solution.mutation_effect)}</td>
                <td>{formatEmergenceClaim(solution.emergence_audit)}</td>
              </tr>
            ))}
            {rows.length === 0 ? (
              <tr>
                <td colSpan={10}>选择包含 tree.json 的 run 后显示 solution tree summary。</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function formatOperatorAssignment(operator: SolutionSummary["operator_assignment"]): string {
  if (!operator?.available) return "root";
  const warning = operator.warning_count ? `, ${operator.warning_count} warn` : "";
  return `${operator.operator_id ?? "operator"} · ${operator.mutation_axis ?? "axis"}${warning}`;
}

function formatKbApplication(kb: SolutionSummary["kb_application"]): string {
  if (!kb?.available) return "none";
  const warnings = kb.warning_count ? `, ${kb.warning_count} warn` : "";
  return `${kb.status ?? "unknown"}${warnings}`;
}

function formatMutationEffect(effect: SolutionSummary["mutation_effect"]): string {
  if (!effect?.available) return "root";
  if (effect.duplicate_of) return `${effect.status ?? "duplicate"} -> ${effect.duplicate_of}`;
  const lines = effect.diff_line_count ?? "n/a";
  return `${effect.status ?? "unknown"} (${lines})`;
}

function formatEmergenceClaim(
  audit: SolutionSummary["emergence_audit"]
): string {
  if (!audit?.available) return "none";
  const gaps = audit.blocking_gap_count ?? 0;
  return gaps ? `${audit.claim_level ?? "unknown"} (${gaps})` : audit.claim_level ?? "unknown";
}

function FigureArtifactsView({ figures }: { figures: ArtifactEntry[] }) {
  if (!figures.length) {
    return <p className="muted">尚未发现 data_overview.svg 或 prediction_overview.svg。</p>;
  }
  return (
    <div className="figure-list">
      {figures.slice(0, 8).map((figure) => (
        <span key={figure.path}>
          <FileText size={14} />
          {figure.path}
        </span>
      ))}
    </div>
  );
}

function formatScore(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) return value.toExponential(3);
  return value.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}

function asRunMode(value: unknown, fallback: RunMode): RunMode {
  return value === "mock" || value === "real" || value === "dry_run" ? value : fallback;
}

function payloadNumber(payload: SolverAction["payload"], key: string, fallback: number) {
  const value = payload?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function payloadStringArray(payload: SolverAction["payload"], key: string, fallback: string[]) {
  const value = payload?.[key];
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : fallback;
}

function payloadRecord(value: unknown, fallback: Record<string, unknown>) {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : fallback;
}

function payloadAgentModels(value: unknown, fallback: Record<string, AgentModelConfig>) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return fallback;
  const result: Record<string, AgentModelConfig> = {};
  for (const [role, rawConfig] of Object.entries(value as Record<string, unknown>)) {
    if (!rawConfig || typeof rawConfig !== "object" || Array.isArray(rawConfig)) continue;
    const config = rawConfig as Record<string, unknown>;
    if (typeof config.model !== "string" || !config.model.trim()) continue;
    result[role] = {
      model: config.model,
      temperature: typeof config.temperature === "number" && Number.isFinite(config.temperature) ? config.temperature : 0,
      reasoning_effort: isReasoningEffort(config.reasoning_effort) ? config.reasoning_effort : undefined
    };
  }
  return Object.keys(result).length ? result : fallback;
}

function defaultAgentModelConfig(role: string, roles: AgentRole[]): AgentModelConfig {
  const settings = roles.find((item) => item.role === role)?.default_model_settings ?? FALLBACK_AGENT_MODEL_SETTINGS;
  return {
    model: "",
    temperature: Number.isFinite(settings.temperature) ? settings.temperature : FALLBACK_AGENT_MODEL_SETTINGS.temperature,
    reasoning_effort: isReasoningEffort(settings.reasoning_effort)
      ? settings.reasoning_effort
      : FALLBACK_AGENT_MODEL_SETTINGS.reasoning_effort
  };
}

function isReasoningEffort(value: unknown): value is ReasoningEffort {
  return value === "low" || value === "medium" || value === "high" || value === "xhigh";
}

function nextStrategyLockId(existing: StrategyLock[]) {
  const used = new Set(existing.map((lock) => lock.lock_id));
  for (let index = 1; index <= existing.length + 100; index += 1) {
    const candidate = `manual_lock_${String(index).padStart(3, "0")}`;
    if (!used.has(candidate)) return candidate;
  }
  return `manual_lock_${Date.now()}`;
}

function strategyLockBranchContext(locks: StrategyLock[]) {
  return {
    expected_inherited_lock_ids: activeStrategyLocks(locks)
      .filter((lock) => lock.scope === "all_branches" && lock.text.trim())
      .map((lock) => lock.lock_id)
  };
}

function activeStrategyLocks(locks: StrategyLock[]) {
  return locks.filter((lock) => lock.text.trim());
}

function normalizeInteger(value: number, fallback: number, min: number) {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(min, Math.floor(value));
}

function AlgorithmCatalog({
  algorithms,
  embedded = false,
  language = "zh",
  onToggle,
  selectedIds = []
}: {
  algorithms: AlgorithmSpec[];
  embedded?: boolean;
  language?: LibraryLanguage;
  onToggle?: (algorithmId: string) => void;
  selectedIds?: string[];
}) {
  const visible = algorithms;
  const isChinese = language === "zh";
  const featureLabel = isChinese ? "特点" : "Features";
  const problemLabel = isChinese ? "对应问题" : "Problem fit";
  const emptyLabel = isChinese ? "算法目录暂未加载。" : "Algorithm catalog is not loaded.";
  const content = (
    <div className="algorithm-grid">
      {visible.map((algorithm) => {
        const selected = selectedIds.includes(algorithm.id);
        const description = textOrFallback(
          isChinese ? algorithm.description_zh : algorithm.description,
          algorithm.description
        );
        const features = stringListOrFallback(
          isChinese ? algorithm.features_zh : algorithm.features,
          [algorithm.description]
        );
        const problemFit = stringListOrFallback(
          isChinese ? algorithm.problem_fit_zh : algorithm.problem_fit,
          algorithm.compatible_benchmark_families
        );
        const safetyNotes = textOrFallback(
          isChinese ? algorithm.safety_notes_zh : algorithm.safety_notes,
          algorithm.safety_notes
        );
        return (
          <article className={selected ? "algorithm-card selected" : "algorithm-card"} key={algorithm.id}>
            <div>
              <span>{algorithm.family}</span>
              <h3>{algorithm.name}</h3>
            </div>
            <p>{description}</p>
            <div className="algorithm-detail-block">
              <span>{featureLabel}</span>
              <ul>
                {features.map((feature) => (
                  <li key={feature}>{feature}</li>
                ))}
              </ul>
            </div>
            <div className="algorithm-detail-block">
              <span>{problemLabel}</span>
              <ul>
                {problemFit.map((problem) => (
                  <li key={problem}>{problem}</li>
                ))}
              </ul>
            </div>
            <div className="algorithm-meta">
              <small>{algorithm.status}</small>
              <small>{algorithm.compatible_benchmark_families.join(" / ")}</small>
              <small>{algorithm.operator?.scheduler_mode ?? "auto-audited"}</small>
              <small>{algorithm.operator?.default_axis ?? "operator"}</small>
            </div>
            {algorithm.implementation_path ? <code>{algorithm.implementation_path}</code> : null}
            <strong>{safetyNotes}</strong>
            {onToggle ? (
              <button type="button" onClick={() => onToggle(algorithm.id)}>
                {selected ? (isChinese ? "已选" : "Selected") : isChinese ? "选择" : "Select"}
              </button>
            ) : null}
          </article>
        );
      })}
      {visible.length === 0 ? <p className="muted">{emptyLabel}</p> : null}
    </div>
  );
  if (embedded) return content;
  return <DataRegion title={isChinese ? "算法目录" : "Algorithm catalog"}>{content}</DataRegion>;
}

function textOrFallback(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function stringListOrFallback(value: unknown, fallback: string[]): string[] {
  if (Array.isArray(value)) {
    const items = value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
    if (items.length > 0) return items;
  }
  return fallback.filter((item) => item.trim().length > 0);
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
        <Metric label="claim_gate" value={metadata?.claim_gate?.status ?? "unknown"} />
        <Metric label="claim_level" value={metadata?.claim_gate?.claim_level ?? "workflow_proxy"} />
        <Metric label="kb" value={metadata?.kb_manifest?.coverage_status ?? "unknown"} />
        <Metric label="multimodal" value={metadata?.multimodal_evidence?.analysis_mode ?? "unknown"} />
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
        <iframe className="vscode-iframe" data-testid="vscode-iframe" src={codeServer.url} title="VS Code Web" />
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
              <tr data-testid={`workspace-row-${workspace.id}`} key={workspace.id} onClick={() => onOpenWorkspace(workspace)}>
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
  onConfirmRealAction,
  onDismissRealAction,
  onResizeReset,
  onResizeStart,
  onResizeStep,
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
  onConfirmRealAction: () => void;
  onDismissRealAction: () => void;
  onResizeReset: () => void;
  onResizeStart: (event: ReactPointerEvent<HTMLButtonElement>) => void;
  onResizeStep: (delta: number) => void;
  onSend: (event: FormEvent) => void;
  onToggle: () => void;
}) {
  if (collapsed) {
    return (
      <aside className="agent-collapsed" data-testid="ide-agent-collapsed">
        <button aria-label="展开 Agent 面板" className="icon-button" type="button" onClick={onToggle}>
          <PanelRightOpen size={18} />
        </button>
        <Bot size={20} />
      </aside>
    );
  }
  return (
    <aside className="agent-panel" aria-label="Agent 面板" data-testid="ide-agent-panel">
      <button
        aria-label="拖动调整 Agent 面板宽度"
        className="agent-resize-handle"
        data-testid="agent-resize-handle"
        type="button"
        onDoubleClick={() => onResizeReset()}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") {
            event.preventDefault();
            onResizeStep(24);
          } else if (event.key === "ArrowRight") {
            event.preventDefault();
            onResizeStep(-24);
          }
        }}
        onPointerDown={onResizeStart}
      />
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
            <button type="button" onClick={onConfirmRealAction}>
              显式确认执行
            </button>
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
          data-testid="ide-agent-input"
          value={message}
          onChange={(event) => onChangeMessage(event.target.value)}
          placeholder="请输入你的问题"
        />
        <button aria-label="发送" data-testid="ide-agent-send" disabled={busy} type="submit">
          <Send size={15} />
        </button>
      </form>
    </aside>
  );
}

function ChatTranscript({ messages }: { messages: AgentMessage[] }) {
  return (
    <div className="chat-transcript" data-testid="chat-transcript">
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
