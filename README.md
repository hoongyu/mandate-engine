# mandate-engine

**The agent proposes; the mandate decides.** A small Python library that sits between an AI agent and every side-effecting tool call (paying an invoice, moving money, sharing data). The model can suggest anything; whether the action runs on its own, waits for a human, runs in shadow mode or never runs is decided outside the model, by a mandate the principal signed.

## What is in it

| Module | Role |
|---|---|
| `engine.mandate` | The mandate: payees saved by the principal, and clauses with an effect (`allow`, `ask`, `observe`, `deny`) and conditions (payee in registry, currency, amount bounds, rolling caps and counts, interest-rate bounds, minimum evidence grade, data-owner consent, trusted arguments). The most restrictive applicable clause wins; the default can never be `allow`. |
| `engine.facts` | Tool-independent facts about one proposed action, so the policy never parses tool arguments. |
| `engine.pdp` | Policy decision point: a pure function from mandate, facts, provenance, history and time to a decision, with a reason for every clause it looked at. |
| `engine.pep` | Policy enforcement point: runs allowed actions, holds the rest for approval, re-decides on the facts at approval time, supports shadow mode and a kill switch. |
| `engine.provenance` | Traces values the agent passes to tools back to where they were seen (mandate, own systems, the user, or untrusted text such as e-mails), so an account number that only appeared in an e-mail is told apart from a saved one. |
| `engine.grade` | Evidence grades and the minimum-grade condition. |
| `engine.audit` | Hash-chained, append-only audit log with JSONL persistence. |
| `engine.identity` | Signed agent credentials a counterparty can check. |
| `engine.history`, `engine.money`, `engine.clock` | Rolling action history, integer minor-unit money, an injectable clock. |
| `engine.agent`, `engine.tools`, `engine.llm` | A minimal agent loop, tool specs, and scripted or LiteLLM model adapters (LiteLLM only loads when a real model is used). |

## Install

```bash
pip install "mandate-engine @ git+https://github.com/hoongyu/mandate-engine@v0.1.0"
# with the LiteLLM adapter:
pip install "mandate-engine[llm] @ git+https://github.com/hoongyu/mandate-engine@v0.1.0"
```

The import package is `engine`. Python 3.12 or newer; the only required dependency is pydantic.

## A minimal decision

```python
from datetime import UTC, datetime, timedelta

from engine.facts import ActionFacts
from engine.history import ActionHistory
from engine.mandate import Clause, Effect, Mandate, Payee
from engine.pdp import decide

now = datetime(2026, 10, 9, tzinfo=UTC)
mandate = Mandate(
    mandate_id="m-1",
    version=1,
    principal="accounts-payable",
    agent_id="ap-agent",
    valid_from=now - timedelta(days=1),
    valid_until=now + timedelta(days=30),
    payees=(
        Payee(
            payee_id="v-102",
            name="Example Supplier GmbH",
            account="DE00 SYNB 0000 0000 0000 01",
            category="supplier",
            verified_via="callback",
        ),
    ),
    clauses=(
        Clause(
            clause_id="P1",
            effect=Effect.ALLOW,
            actions=frozenset({"pay_invoice"}),
            payee_in_registry=True,
            currencies=frozenset({"EUR"}),
            max_amount_minor=2_000_000,  # EUR 20,000.00
        ),
    ),
)

small = ActionFacts(action="pay_invoice", payee_id="v-102", currency="EUR", amount_minor=860_000)
large = small.model_copy(update={"amount_minor": 5_000_000})
for facts in (small, large):
    d = decide(mandate, facts, labels={}, history=ActionHistory(), now=now)
    print(facts.amount_minor, d.effect, d.reasons)
```

The EUR 8,600 payment is allowed under clause P1; the EUR 50,000 one falls outside P1, so the default effect applies and a human is asked. Every decision carries the reason for each clause.

## Where this code comes from

This repository was extracted on 2026-10-09 from the author's private repository (an entry to FinTechathon 2026), keeping the full history of `engine/` and its unit tests: 21 commits dated 2026-09-22 to 2026-10-05. Commit dates are unchanged; the author e-mail was rewritten to the GitHub no-reply address.

The code is unchanged. The git tree of `engine/` here, `b23fb9b0615e8a9c0fef719996cafb98ccd7a80b`, is identical to `engine/` at commit `a0dfb55` (2026-10-05 23:43 +02:00) of the private repository. The only commit added after extraction is the packaging (this README, `pyproject.toml`, `LICENSE`, `NOTICE`, `.gitignore`, and a two-line `tests/conftest.py` that keeps LiteLLM from fetching its price map at import).

Hackathon entries that build on this engine declare it as a pre-existing component; it is not part of their new work.

## Tests

```bash
uv sync --group dev --extra llm   # the LiteLLM adapter test needs the optional extra
uv run pytest
```

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
