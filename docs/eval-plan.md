# Evaluation plan

## Aim

I will compare the retrieval methods and agent variants using the same questions and stored source material. The main questions are whether graph retrieval helps with linked facts and changed facts, and whether the agent's additional steps improve the answers enough to justify their cost.

## Configurations

| Name | Intended setup |
| --- | --- |
| external | Independent LlamaIndex text-retrieval baseline |
| classic | Chroma text retrieval with a direct answer |
| graphiti | Graphiti hybrid retrieval with a direct answer |
| own | Custom graph retrieval with a direct answer |
| own_agents | Custom graph retrieval with bubble investigations |
| classic_agents | Text retrieval with the same controller and text tools |

The development runner exercises all six labels using scripted components. Genuine backend comparisons start after the adapters are connected. The external development label is not a LlamaIndex implementation.

## Question set

The development snapshot has ten fictional source items and twelve questions, with three examples in each category below. These questions stay outside the final set.

| Final category | Planned count |
| --- | --- |
| Single-bubble factual | 15 |
| Multi-hop across bubbles | 20 |
| Time-sensitive changed facts | 15 |
| Freshness | 10 |
| Total | 60 |

Each question needs a reference answer, relevant bubbles, a time scope and the source assertions required to answer it. Multi-hop labels must include the intermediate facts. Equivalent passages and graph facts can map to the same assertion. Another reviewer should check the labels before the set is frozen.

## Shared conditions

I will use the same eligible source content, embedding model and answer model across the integrated runs. Retrieval parameters, prompts, context size, model settings and source cutoffs will be recorded. Agent variants share the action and time budgets. Graph agents have additional graph tools, so a difference between agent variants cannot automatically be attributed to graph topology alone.

Inputs are frozen before a controlled run. The loader checks the data hashes, and the runner records code hashes, execution order and per-question outputs. Gold answers and evidence labels stay on the scoring side; answering adapters receive the question and query settings.

## Measures

| Measure | Definition | Planned target |
| --- | --- | --- |
| Recall@10 | Required assertions covered by the first ten unique evidence units | Graph retrieval exceeds classic on multi-hop questions |
| MRR and Hits@1 | Rank of the first relevant unit and relevance of the first unit | Report by configuration and category |
| Query routing | Precision and recall of selected bubbles | Recall at least 0.90; precision at least 0.70 |
| Ingestion routing | Exact agreement with independently labeled bubble sets | At least 0.85 on 50 items |
| Answer correctness | 0 wrong, 1 partly correct, 2 complete and supported | Full system improves multi-hop mean by at least 0.30 points over classic and external separately |
| Citation faithfulness | Independently judged support for claim/citation links | At least 0.90 |
| Changed-fact accuracy | Correct answers on changed-fact questions | At least 0.80 |
| Freshness lag | Publication to first retrieval visibility | No more than poll interval plus five minutes |
| Duplicate entities | Extra duplicate records within a bubble | At most 5 percent in a 100-entity audit |
| Ungrounded entities | Extracted entities unsupported by stored text | Report |
| Memory | Combined application and VM use without counting containers twice | At most 4 GB |

Citation-reference validity is a structural check, separate from whether the cited passage supports the claim. Missing judgments remain unmeasured. An unanswerable question has no retrieval-recall denominator; failed answerable questions remain in the run.

## Time and updates

The query has two time settings: when a fact is valid, and what information was available to the system. Validity intervals include their start and exclude their end. Later changes must not leak into earlier knowledge states. Adapters need versioned records or replayed states when a final index cannot reconstruct the earlier state.

The 50 non-freshness questions use frozen inputs. The ten freshness cases will replay the same arrivals into reset stores and record when each item first becomes searchable. The existing development timestamps test filtering only; live arrival-to-visibility measurements remain pending.

## Judging and analysis

Correctness and citation labels will be assigned independently of the system configuration. The scoring command imports identified ratings and leaves incomplete citation judgments unscored. A judge model, if used, must be checked against human ratings before its scores are accepted.

Results will be reported by question category. Comparisons use paired questions, with mean differences and uncertainty intervals. Call counts, context length, token use and latency should be reported alongside quality. A positive development check does not establish answer-quality gains.

## Remaining work

The evaluation code and development data are ready for integration checks. The remaining work is to connect genuine backends, collect the final corpus, review the question labels, calibrate judgments, run the comparisons, and measure live updates and memory use.
