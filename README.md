# OmniStreamRAG agent and evaluation

Version 0.2.0 contains the agent controller, evaluation runner and development tests for OmniStreamRAG. The implemented modules cover bubble selection, bounded investigations, evidence handling, citation checks and evaluation exports.

## Agent controller

The controller selects relevant bubbles and supports direct retrieval or parallel investigations for multi-hop questions. Each investigation has a default budget of six action attempts and 60 seconds. Text retrieval exposes search and stored-item access; graph retrieval also exposes related-entity and fact requests.

Returned evidence is checked against bubble exclusions, time constraints and grounding fields. Partial results survive individual investigation failures. Source conflicts remain visible, and each answer claim must reference retrieved evidence.

Model and retrieval services connect through the interfaces in `src/omnistream/contracts.py`. The current development implementation uses scripted model decisions and a small lexical store. Live model, Chroma, Graphiti, Neo4j and LlamaIndex adapters remain to be integrated.

## Evaluation

The runner supports six configurations: `external`, `classic`, `graphiti`, `own`, `own_agents` and `classic_agents`. The development snapshot contains ten fictional source items and twelve questions covering factual, multi-hop, temporal and freshness cases.

Implemented measures include Recall@10, MRR, Hits@1 and bubble-selection precision and recall. CSV exports contain per-question results and configuration summaries; JSON traces retain the selected route, evidence and answer. Separate rating imports support answer correctness and citation faithfulness.

All six development configurations currently share the lexical store. These runs verify controller and evaluation behavior. Comparisons between retrieval methods and answer-quality measurements remain pending until integration and independent judging are complete.

## Verification

The saved verification run passed **53 automated tests** and completed **72 development runs**. The log, result tables and traces are in `verification/`. The automated workflow is defined in `.github/workflows/verify.yml`.

Python 3.12 or later is required. The development check uses the Python standard library and requires no external service credentials.

| Environment | Verification command |
| --- | --- |
| Windows PowerShell with Python 3.12 | `py -3.12 verify.py` |
| macOS or Linux Terminal | `python3 verify.py` |

Saved output is available through `--output results/run-01`; the destination must be new or empty. The `--verbose` option lists individual tests.

## Repository contents

| Path | Contents |
| --- | --- |
| `src/omnistream/agents.py` | Agent controller and evidence handling |
| `src/omnistream/contracts.py` | Model, retrieval, query and evidence interfaces |
| `src/omnistream/evaluation.py` | Evaluation runner, exports and rating imports |
| `src/omnistream/metrics.py` | Retrieval, routing and supporting metric functions |
| `src/omnistream/fixtures.py` | Scripted development adapters |
| `data/pilot_snapshot/` | Fictional source items, questions and relevance labels |
| `tests/` | Controller, contract, temporal and evaluation tests |
| `verification/` | Saved verification results |
| `docs/` | Evaluation plan and integration contracts |

## Next stage

The next stage is module integration followed by evaluation on a separate 60-question set. Remaining measurements include answer correctness, citation support, live-update delay and resource use.

Technical details are recorded in the [evaluation plan](docs/eval-plan.md), [integration contracts](docs/integration.md) and [model adapter contract](docs/prompts.md).

