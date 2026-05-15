# Budgeted PINN Mutation

Applies to: `burgers_pinn_faithful_small`.

Sources:

- Public code path: `maziarraissi/PINNs/appendix/continuous_time_inference (Burgers)/Burgers_systematic.py`.
- OpenAI Agents SDK official docs: Agent definitions, Orchestration and handoffs, Guardrails and human review, Integrations and observability.

Use the public systematic script as a source of experiment dimensions, not as a
runtime dependency. The dimensions worth exposing to the agent are supervised
sample count, IC/BC fit, collocation count, network depth, network width,
residual weight, and training schedule. In AgenticSciML, the Python
orchestrator keeps control of state, score, files, and champion selection,
while the agent proposes one bounded mutation with structured output.

Good proposals:

- change only one or two budget dimensions per mutation;
- state the expected effect, failure mode, and rollback signal;
- prefer traceable small experiments before higher-cost model expansions.

SDK-style guardrails:

- output a typed implementation plan rather than free-form control transfer;
- keep evaluator and artifact writes in trusted Python code;
- record the retrieved KB entry, prompt, response, score, and guardrail outcome
  under the run directory.
