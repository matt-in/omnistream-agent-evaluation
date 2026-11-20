# Module integration

## Application and evaluation entry points

`ProjectSystem.ask` uses custom graph retrieval with subagents enabled for multi-hop queries. `ProjectSystem.answer` runs the specified evaluation configuration. Chroma and Graphiti search use separate retrieval adapters, and the independent LlamaIndex baseline has its own answering adapter.

| Interface | Contract |
| --- | --- |
| `Model.route` | Bubble identifiers, query strategy and resolved time scope |
| `Model.step` | One tool request or finish action |
| `Model.synthesize` | Claims with evidence identifiers or an abstention |
| `Backend.execute` | Ranked evidence for one scoped tool request |
| `SourceStore.clean_text` | Immutable clean text by item identifier |
| `SystemAdapter.answer` | Answer and ranked evidence identifiers for an evaluation case |

The project uses local embeddings and the shared language-model provider through the model adapter. The custom retrieval adapter owns entity seeding, graph expansion through global entities, personalized PageRank and reciprocal rank fusion. The ranking favors relevance with intended relative weights of 2 for relevance, 1 for recency and 1 for source support. This package consumes the ranked results through the adapter interface.

## Scope and provenance

Direct lookup uses search and can read a matching named item. Multi-hop investigations use the same defaults of six attempts and 60 seconds. The text mode permits search and stored-item reads. Graph modes also permit neighbor requests up to two hops and fact requests for a specified time window. Graph requests use discovered entity identifiers. The custom graph adapter checks every cross-bubble link against the selected bubbles and exclusions.

Adapters translate team records to the evidence contract. Fact and chunk identifiers must be stable across snapshots and match the relevance labels. Graphiti groups and Chroma collections must map to the same bubble catalog. Each fact carries a source item, publication time, availability time, grounding flag and the validity and invalidation times when applicable. Canonical entity identifiers support cross-bubble joins and candidate conflict checks.

Offsets reference Unicode characters in the stored clean text. A paraphrased fact carries the corresponding original passage as its evidence text. The controller checks the passage at those offsets through `SourceStore`; independent judgment determines claim support. The application can use each citation's item identifier, offsets and source URL for its source display.

Fact validity and knowledge availability are separate. The router adapter resolves missing years and relative dates such as questions about the period before an event. An explicit query window takes precedence. Retrieval adapters apply time and scope filtering before ranking. Historical facts can be returned when their validity overlaps the requested scope and they were available at the knowledge cutoff.

## Evaluation contract

The runner supplies question inputs and configuration settings. Reference answers and relevance labels remain on the scoring side. `Answer.ranked_evidence_ids` identifies the retrieval ranking separately from the answer context. A controller run interleaves its bubble rankings, retaining first occurrences and excluding original-item reads. Cross-configuration comparisons require equivalent definitions for the external ranking and a consistent mapping of chunks and facts to supporting assertions.

Adapters use `ProviderUnavailable` for retryable service failures and `ProviderBlocked` for credit or authorization failures. The checkpoint identity includes the snapshot, code hash, configurations, seed, timeout and adapter revision. The revision identifies the fixed service settings and snapshot version used by the adapter. One process operates on each run directory.

Both the standalone package and shared application use the `omnistream` namespace. The standalone package runs in its own checkout and Python environment. Incorporation into the shared application requires explicit interface adapters and an integration check against the original store, graph and text indexes. Cross-bubble, historical and provider-failure cases follow the initial single-item check.
