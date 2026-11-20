# OmniStreamRAG agent and evaluation

Version 0.3.0 contains the query controller and evaluation tools for the first final-report draft. The controller and runner have been verified with local development components. The adapters for the shared application and its live services are pending.

## Query controller

The application entry point uses the custom graph retriever. A direct question uses search, with original text available for a named item. Multi-hop queries can start one investigation per selected bubble. An investigation has a default budget of six action attempts and 60 seconds. The text baseline exposes search and item reads. Graph modes also expose related entities and dated facts.

The controller preserves pin and exclusion settings, filters returned evidence by time and scope, and retains partial findings within the query budget. When the original store is connected, a citation carries its stored text offsets and the result of an exact span check. Separate judgments measure whether each passage supports its claim.

## Evaluation

The runner supports the custom graph and Chroma with and without subagents, Graphiti search, and the independent LlamaIndex baseline. The pilot contains ten fictional source items and twelve questions. Scripted adapters share a lexical store for this check.

Completed cases are saved to a SQLite checkpoint. A resumed run preserves measured cases and their ratings. Service failures are recorded as unmeasured. Credit and authorization failures stop the run. The outputs record executed, unmeasured and pending cases, recall and rank metrics, answer traces and independent ratings. Paired correctness comparisons use the questions scored in both configurations.

The team evaluation uses the 60-question synthetic eval-v2 set and the real-v2 development and held-out splits. The plain Python package includes the pilot snapshot. Live benchmark execution requires the team's frozen data, evidence mapping, model and retrieval services.

## Verification

The saved verification passed **63 automated tests** and completed **72 pilot cases** across the six configurations. The test log and result files are in [verification](verification/). The GitHub Actions workflow runs the same checks with Python 3.12.

| Environment | Command from the repository root |
| --- | --- |
| Windows PowerShell with Python 3.12 | `py -3.12 verify.py` |
| macOS or Linux with Python 3.12 or later | `python3 verify.py` |

The development check uses the Python standard library. The `--output results/run-01` option saves CSV results and traces to a new or empty directory. The `--verbose` option lists the individual tests.

| Path | Contents |
| --- | --- |
| `src/omnistream/agents.py` | Investigation loop, evidence and citations |
| `src/omnistream/integration.py` | Application and evaluation adapter selection |
| `src/omnistream/evaluation.py` | Snapshot validation, execution, metrics and exports |
| `src/omnistream/checkpoints.py` | Checkpoint persistence and run identity |
| `data/pilot_snapshot/` | Development questions, sources and relevance labels |
| `tests/` | Controller and evaluation tests |

The [integration contracts](docs/integration.md), [evaluation plan](docs/eval-plan.md) and [model adapter contract](docs/prompts.md) describe the inputs and expected behavior.
