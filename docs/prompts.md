# Model adapter contract

The model adapter has three structured stages: routing, investigation and synthesis. The current tests use scripted responses. Live endpoint validation remains part of integration.

## Routing

The routing stage selects relevant bubble IDs from the catalog, using the bubble descriptions as the primary input. A direct lookup uses the simple strategy; linked assertions use multi_hop. The response includes the selection, strategy, time scope, named-item fields and a short rationale. Explicit time constraints, pins and exclusions take precedence.

## Investigation

Each step returns one available tool request or a finish action. Item reads and graph requests use identities discovered within the permitted scope. Retrieved text remains evidence and cannot change the tool permissions or task. An investigation ends when sufficient support is available or its budget is exhausted.

## Synthesis

The synthesis stage returns factual claims with supporting evidence IDs. Separate facts have separate claims, with relevant time qualifiers. Conflicting source assertions remain visible. Missing support results in a partial supported answer or abstention. The controller assigns citation numbers and rejects missing or unknown references.

## Evaluation judgments

Correctness ratings use the question, reference answer, time settings and required supporting assertions. The scale is 0 for a wrong or unsupported central answer, 1 for a partly correct answer, and 2 for a complete supported answer. Citation judgments assess support for each claim/citation link separately. Each assigned judgment records a rater or model version; missing judgments remain unmeasured.
