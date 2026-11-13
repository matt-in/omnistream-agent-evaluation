# OmniStreamRAG agent and evaluation

My work covers the query-time agent controller and the evaluation tools for OmniStreamRAG. The controller handles bubble selection, tool requests, evidence checks and cited responses. The evaluation runner records results for six configurations and exports them for comparison.

This version runs with scripted model and retrieval components. Live model, Chroma, Graphiti, Neo4j and LlamaIndex connections are still to be added. The development results check the code and workflow; answer quality will be assessed after integration.

## Check on GitHub

The included workflow runs the checks on GitHub after the files are committed. No local Python installation is needed for this option.

1. Extract the ZIP and upload the project contents to a repository.
2. Make sure `.github/workflows/verify.yml` is included.
3. Open **Actions → Verify agent and evaluation** and open the latest run.

The current expected result is **53 passing tests and 72 completed development runs**. Full upload steps for Chrome on Windows and for Mac are in [UPLOAD_TO_GITHUB.md](UPLOAD_TO_GITHUB.md).

## Check on Windows

Use Python 3.12. Open the extracted project folder in PowerShell and run:

```powershell
py -3.12 verify.py
```

To save CSV files and answer traces:

```powershell
py -3.12 verify.py --output results/my-check
```

## Check on Mac or Linux

Use Python 3.12 or later. In Terminal, change to the extracted project folder and run:

```bash
python3 verify.py
```

To save the outputs:

```bash
python3 verify.py --output results/my-check
```

Use a new output folder for each saved run. Add `--verbose` to list the individual tests. No API key, database or additional Python package is needed for the development check.

## Main files

| File | Purpose |
| --- | --- |
| `src/omnistream/agents.py` | Agent loop, limits, evidence handling and citation checks |
| `src/omnistream/contracts.py` | Shared query, model, tool and evidence interfaces |
| `src/omnistream/evaluation.py` | Evaluation runs, CSV exports and rating imports |
| `src/omnistream/metrics.py` | Retrieval, routing, citation and timing metrics |
| `src/omnistream/fixtures.py` | Scripted components for development |
| `tests/` | Automated checks |
| `data/pilot_snapshot/` | Ten fictional items, twelve questions and evidence labels |
| `verification/` | Saved results from the completed development check |

The test configurations share the same small lexical store. Their results are not a comparison of the genuine retrieval backends. Correctness and citation-faithfulness scores stay blank until separate judgments are entered.

The final evaluation is planned for 60 questions. The corpus, independent labels, live-update checks and hardware measurements remain to be completed.

Further details: [evaluation plan](docs/eval-plan.md), [module integration](docs/integration.md), and [model instructions](docs/prompts.md).
