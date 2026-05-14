from __future__ import annotations

import json
from typing import Any

from agenticsciml.agents.base import AgentBase
from agenticsciml.agents.critic import CriticAgent
from agenticsciml.agents.specs import PromptTemplate
from agenticsciml.state import AgentMessage, Proposal


class ProposerAgent(AgentBase):
    role = "proposer"

    def debate(
        self,
        solution_id: str,
        parent_summary: str,
        kb_entry: str | None,
        related_reports: list[str],
        use_critic: bool = True,
        branch_context: dict[str, Any] | None = None,
    ) -> Proposal:
        self.require_inputs(
            {
                "solution_id": solution_id,
                "parent_summary": parent_summary,
                "kb_entry": kb_entry,
                "related_reports": related_reports,
                "branch_context": branch_context,
            }
        )
        messages: list[AgentMessage] = []
        critic_feedback: list[str] = []
        context = (
            f"Parent summary:\n{parent_summary}\n\n"
            f"KB entry:\n{kb_entry or 'none'}\n\n"
            f"Related reports:\n{chr(10).join(related_reports) if related_reports else 'none'}\n\n"
            "Branch context:\n"
            f"{json.dumps(branch_context or {}, indent=2, sort_keys=True)}"
        )
        critic = CriticAgent(self.llm, self.storage) if use_critic else None
        proposal_hint = "No proposal yet; critique the diagnostic framing."
        for round_index in range(1, 5):
            if round_index < 3:
                prompt = PromptTemplate(
                    "Proposer round {round_index}: provide concise diagnostic reasoning summary. "
                    "Do not reveal hidden chain-of-thought and do not propose implementation yet.\n\n"
                    "{context}"
                ).render({"round_index": round_index, "context": context})
                response = self.complete_text(prompt)
                proposal_hint = response
            elif round_index == 3:
                prompt = PromptTemplate(
                    "Proposer round 3: synthesize a concrete implementation plan. "
                    "Use concise rationale summaries only.\n\n"
                    "{context}"
                ).render({"context": context})
                response = self.complete_text(prompt)
                proposal_hint = response
            else:
                critic_context = "\n".join(f"- {item}" for item in critic_feedback) or "No critic feedback."
                prompt = PromptTemplate(
                    "Proposer round 4: return final implementation-ready proposal as JSON "
                    "with title, diagnosis, mutation_plan, expected_effect, risks. "
                    "`mutation_plan` and `risks` must be JSON arrays of strings, not strings.\n\n"
                    "{context}\n\nCritic feedback to address:\n{critic_feedback}"
                ).render({"context": context, "critic_feedback": critic_context})
                data = self.complete_json_checked(
                    prompt,
                    "proposal",
                    required_fields=("title", "diagnosis", "mutation_plan", "expected_effect", "risks"),
                )
                response = str(data)
                proposal = Proposal.from_dict(data)
            messages.append(AgentMessage(self.role, prompt, response, {"round": round_index}))

            if round_index < 4 and critic is not None:
                critic_response = critic.critique(
                    solution_id=solution_id,
                    proposal_summary=proposal_hint,
                    context=context,
                    round_index=round_index,
                )
                critic_feedback.append(critic_response)
                messages.append(
                    AgentMessage("critic", f"See critic_round_{round_index}.json", critic_response, {"round": round_index})
                )

        self.storage.save_solution_text(solution_id, "proposal.md", proposal.to_markdown())
        if critic is None:
            self.storage.save_solution_text(
                solution_id,
                "critic.md",
                "# Critic Disabled\n\nThis ablation run skipped CriticAgent calls.\n",
            )
        self.require_artifacts(solution_id, ("proposal.md", "critic.md"))
        self._save_messages(solution_id, messages, "proposal_debate")
        return proposal
