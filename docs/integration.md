# Module integration

## Interfaces

The controller depends on a model adapter, retrieval adapters and a bubble catalog. The data types and async interfaces are in `src/omnistream/contracts.py`.

| Interface | Required behavior |
| --- | --- |
| Model.route | Select bubble IDs, strategy and time scope |
| Model.step | Return one structured tool request or finish |
| Model.synthesize | Return claims with retrieved evidence IDs, or abstain |
| Backend.execute | Return ranked evidence for a scoped tool request |
| SystemAdapter.answer | Run the requested configuration and return an Answer |

The built-in fixtures provide scripted model behavior and lexical retrieval for development. They are replaced when the live services are connected.

## Tool requests

| Tool | Arguments | Bound |
| --- | --- | --- |
| search | query | Top 10 text records or top 20 graph records |
| neighbors | entity_id, depth | Discovered identity and depth at most two |
| facts | entity_id, from, to | Discovered identity and valid time range |
| read_item | item_id, parts | Discovered item and at most three requested parts |

Classic mode exposes search and read_item. Other modes also expose neighbors and facts. The controller limits an investigation to six action attempts and 60 seconds by default. Invalid requests use an attempt. Backend clients need their own request timeouts and must support cancellation.

`Backend.execute` receives the active bubble, mode, tool request, time scope, knowledge cutoff and `allowed_bubbles`. Text and Graphiti workers stay in their active bubble. Custom graph retrieval may cross into other selected bubbles, but not excluded ones.

The custom retrieval adapter owns entity seeding, traversal and ranking. The controller does not implement PageRank or database queries. It checks results before merging them and bounds the final answer context.

## Evidence

An evidence record identifies the returned unit, stored item, bubble and source passage. Its start and end are zero-based Unicode character offsets into immutable clean text. Confirm that the stored text at those offsets matches the passage. Graph paraphrases need a supporting source passage.

`published_at` is source publication time. `available_at` is first visibility in the relevant retrieval backend. `valid_from` and `valid_to` describe world time. `invalidated_at` records when an invalidation became known. Apply eligibility filters before ranking; the controller's second check cannot recover relevant records already displaced by an ineligible top-k result.

Optional canonical subject, relation and object IDs support conflict checks. Global identities and paths support cross-bubble explanations. Multi-valued relations need a cardinality policy before treating every different object as a contradiction.

## Answering and evaluation

Validate structured model output before constructing the contract objects. Keep explicitly supplied query settings authoritative. A reference that points to retrieved evidence passes the structural citation check; semantic support still needs separate judgment.

The external evaluation adapter should call the independent LlamaIndex pipeline. The other five configurations use the controller and their relevant retrieval adapters. Adapters must not read reference answers or evidence labels when answering.

Start integration with one item in one bubble. Check search, original-text access and citation positions. Then check a cross-bubble question, an exclusion, a changed fact and a partial failure. Run all six genuine configurations only after these paths work consistently.
