# Evaluation plan

## Configurations

| Configuration | Retrieval and answering |
| --- | --- |
| `external` | Independent LlamaIndex pipeline |
| `classic` | Chroma chunks with direct answering |
| `graphiti` | Graphiti search with direct answering |
| `own` | Custom graph retrieval with direct answering |
| `own_agents` | Custom graph retrieval with subagents on multi-hop questions |
| `classic_agents` | Chroma with the same subagent budgets and text tools |

The primary comparisons pair custom graph retrieval with subagents against classic and external on multi-hop questions. Custom graph plus subagents versus classic plus subagents measures the difference between the two configured systems under an agent loop. Comparisons within each retrieval mode estimate the effect of enabling the loop. Graph configurations expose additional tools, so attributing this difference to graph topology alone requires a separate control.

## Datasets

| Set | Items and questions | Purpose |
| --- | --- | --- |
| Pilot | 10 fictional items and 12 questions | Local controller and runner verification |
| eval-v2 | 90 fictional items and 60 questions | Synthetic development and retrieval diagnosis |
| real-v2 development | 24 scored questions | Development with recorded source items |
| real-v2 held-out | 55 scored questions | Final paired comparison |

The eval-v2 mix is 15 factual, 20 multi-hop, 15 temporal and 10 freshness questions. The real-v2 corpus contains 350 items. The development and held-out questions use disjoint supporting source items. Synthetic examples, development changes and judge calibration remain separate from the held-out run.

The executable pilot has `items.jsonl`, `questions.jsonl`, `qrels.json` and a hash manifest. The normalized question data records a question identifier, category, text, knowledge cutoff, answerability, expected bubbles and supporting assertion identifiers. An optional `scope` is an explicit input to the query. The route resolves the window for questions that test date interpretation. Relevance labels map returned chunk and fact identifiers to common support identifiers.

## Measurement

| Measure | Definition and target |
| --- | --- |
| Recall@10 | Fraction of the required assertions covered by the first ten unique retrieval units. Graph modes are expected to exceed classic on multi-hop cases. |
| MRR and Hits@1 | Reciprocal rank of the first relevant unit and relevance of the first unit. Both are reported by configuration. |
| Query routing | Set precision and recall for selected bubbles, with targets of 0.70 and 0.90 respectively. |
| Answer correctness | Independent 0, 1 or 2 rating. The full system targets a mean improvement of at least 0.30 points over classic and external on multi-hop cases. |
| Citation faithfulness | Fraction of claim-citation links judged to support the claim, with a target of 0.90. |
| Changed facts | Correctness on changed-fact questions, with a target of 0.80. |
| Freshness | Publication to retrieval visibility, with a target of the polling interval plus five minutes. Injection-to-answer delay is a separate measure. |

Ingestion exact-set routing accuracy, entity duplication, grounding, total memory and provider cost require observations from the shared application. The targets are at least 0.85 for ingestion routing accuracy, at most 5% duplicates in a 100-entity audit and at most 4 GB of peak combined memory.

Shared conditions include a frozen corpus, consistent eligibility and time cutoffs, and recorded model, embedding, ranking and context settings. Freshness replays use the same item arrivals and reset stores. The first retrievable time is recorded separately from publication and injection.

## Execution and scoring

The runner saves each outcome with its question and configuration. Unmeasured cases retain failure statuses and have blank scores. Missing and pending cases are counted separately. Valid abstentions and empty retrievals remain measured outcomes. Resumed execution reuses the same data and settings and skips measured cases.

Ratings are assigned independently of the configuration. Answer correctness grades are 0 for wrong or unsupported, 1 for partly correct and 2 for complete and supported. Citation-reference and source-offset checks are separate from semantic support ratings. Automatic judgments require calibration against a blinded human sample. Scored exports include paired differences on shared question IDs and the number of measured pairs. Final comparisons depend on coverage, documented settings and the held-out results.
