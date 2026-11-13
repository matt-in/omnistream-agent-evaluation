"""Bounded asynchronous controller independent of Graphiti, Chroma and providers."""

import asyncio
import math
from dataclasses import dataclass, replace
from itertools import zip_longest
from typing import Any

from .contracts import (
    Backend, Bubble, Claim, Configuration, Draft, Evidence, Finish, Model,
    Observation, Query, Route, TimeScope, ToolCall, timestamp,
)


@dataclass(frozen=True)
class Limits:
    tool_calls: int = 6
    agent_seconds: float = 60
    router_seconds: float = 30
    synthesis_seconds: float = 60
    max_context_chars: int = 24000
    max_context_units: int = 60
    part_chars: int = 8000

    def __post_init__(self):
        for name in ("tool_calls", "max_context_chars", "max_context_units", "part_chars"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("agent_seconds", "router_seconds", "synthesis_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")


@dataclass
class Investigation:
    bubble_id: str
    status: str
    evidence: tuple[Evidence, ...]
    history: tuple[Observation, ...]


@dataclass
class Answer:
    status: str
    route: Route | None
    investigations: tuple[Investigation, ...]
    evidence: tuple[Evidence, ...]
    claims: tuple[Claim, ...]
    conflicts: tuple[dict, ...]
    citations: tuple[dict, ...]
    text: str
    errors: tuple[str, ...] = ()


def overlap(a: Evidence, b: Evidence) -> bool:
    lows = [timestamp(x) for x in (a.valid_from, b.valid_from) if x]
    highs = [timestamp(x) for x in (a.valid_to, b.valid_to) if x]
    return not lows or not highs or max(lows) < min(highs)


def merge_conflicts(evidence: tuple[Evidence, ...]) -> tuple[dict, ...]:
    """Report candidate conflicts; multi-valued relations need schema review."""
    groups: dict[tuple, list[Evidence]] = {}
    for fact in evidence:
        if fact.subject and fact.relation and fact.object:
            groups.setdefault((fact.subject, fact.relation), []).append(fact)
    result = []
    for (subject, relation), facts in sorted(groups.items()):
        conflicting = {
            f.unit_id: f for f in facts
            if any(f.object != g.object and overlap(f, g) for g in facts)
        }
        if not conflicting:
            continue
        eligible = [f for f in conflicting.values() if f.credibility >= 0.6]
        eligible.sort(key=lambda f: (timestamp(f.published_at), f.unit_id), reverse=True)
        preferred = None
        if eligible:
            newest = timestamp(eligible[0].published_at)
            tied = [f for f in eligible if timestamp(f.published_at) == newest]
            if len({f.object for f in tied}) == 1:
                preferred = eligible[0].unit_id
        result.append({
            "subject": subject, "relation": relation,
            "evidence_ids": sorted(conflicting), "preferred": preferred,
            "reason": "latest_assertion_with_credibility_at_least_0.6" if preferred else "unresolved",
        })
    return tuple(result)


class AgentRunner:
    def __init__(self, model: Model, backend: Backend, catalog: tuple[Bubble, ...],
                 limits: Limits | None = None):
        self.model, self.backend, self.catalog = model, backend, catalog
        self.limits = limits or Limits()
        if len({b.id for b in catalog}) != len(catalog):
            raise ValueError("Bubble IDs must be unique")

    async def run(self, query: Query, configuration: Configuration) -> Answer:
        if configuration.mode == "external":
            raise ValueError("external requires a LlamaIndex adapter")
        ids = {b.id for b in self.catalog}
        if (set(query.pin) | set(query.exclude)) - ids:
            raise ValueError("Pin or exclusion names an unknown bubble")
        if set(query.pin) & set(query.exclude):
            raise ValueError("A bubble cannot be pinned and excluded together")
        try:
            route = await asyncio.wait_for(
                self.model.route(query, self.catalog), self.limits.router_seconds,
            )
            if not isinstance(route, Route) or route.strategy not in ("simple", "multi_hop"):
                raise ValueError("Invalid route schema")
            if set(route.bubbles) - ids:
                raise ValueError("Route names an unknown bubble")
            if not isinstance(route.scope, TimeScope):
                raise ValueError("Route needs a valid time scope")

            selected = (set(route.bubbles) | set(query.pin)) - set(query.exclude)
            bubbles = tuple(b for b in self.catalog if b.id in selected)
            scope = query.scope or route.scope
            query = replace(query, scope=scope)
            route = Route(tuple(b.id for b in bubbles), route.strategy, scope,
                          route.names_specific_item, route.item_id, route.rationale)
        except Exception as error:
            return self._empty("routing_failed", errors=(type(error).__name__,))
        if not bubbles:
            return self._empty("no_bubbles", route)
        use_agents = configuration.subagents and route.strategy == "multi_hop"
        investigations = tuple(await asyncio.gather(*(
            self._investigate(query, bubble, configuration, use_agents, route)
            for bubble in bubbles
        )))
        evidence = self._context(investigations)
        conflicts = merge_conflicts(evidence)
        if not evidence:
            return Answer("no_evidence", route, investigations, (), (), (), (),
                          "Insufficient evidence in the selected sources.")
        try:
            draft = await asyncio.wait_for(
                self.model.synthesize(query, evidence, conflicts), self.limits.synthesis_seconds,
            )
            if not isinstance(draft, Draft):
                raise ValueError("Invalid answer schema")
            if draft.abstain or not draft.claims:
                return Answer("abstained", route, investigations, evidence, (), conflicts, (),
                              "Insufficient evidence to answer the question.")
            allowed = {e.unit_id: e for e in evidence}
            for claim in draft.claims:
                if not isinstance(claim, Claim) or not claim.text.strip() or not claim.evidence_ids:
                    raise ValueError("Each factual claim needs evidence")
                if set(claim.evidence_ids) - allowed.keys():
                    raise ValueError("Citation points outside retrieved evidence")
            cited_ids = list(dict.fromkeys(u for c in draft.claims for u in c.evidence_ids))

            cited_ids.extend(u for c in conflicts for u in c["evidence_ids"] if u not in cited_ids)
            citations = tuple({"number": i + 1, "unit_id": u,
                               "item_id": allowed[u].item_id, "start": allowed[u].start,
                               "end": allowed[u].end, "bubble_id": allowed[u].bubble_id}
                              for i, u in enumerate(cited_ids))
            numbers = {c["unit_id"]: c["number"] for c in citations}
            text = "\n".join(c.text + " " + "".join(f"[{numbers[u]}]" for u in dict.fromkeys(c.evidence_ids))
                             for c in draft.claims)
            displayed_claims = list(draft.claims)
            for conflict in conflicts:
                sides = "; ".join(f"{allowed[u].text} [{numbers[u]}]" for u in conflict["evidence_ids"])
                displayed_claims.extend(Claim("Source assertion: " + allowed[u].text, (u,))
                                        for u in conflict["evidence_ids"])
                pref = conflict["preferred"]
                text += "\nConflicting source assertions: " + sides
                text += f". Preferred assertion: [{numbers[pref]}]." if pref else ". Preference unresolved."
            partial = any(i.status not in ("complete", "direct") for i in investigations)
            return Answer("partial" if partial else "ok", route, investigations, evidence,
                          tuple(displayed_claims), conflicts, citations, text)
        except Exception as error:
            return Answer("synthesis_failed", route, investigations, evidence, (), conflicts, (),
                          "Answer could not be validated.", (type(error).__name__,))

    @staticmethod
    def _empty(status: str, route: Route | None = None, errors=()) -> Answer:
        return Answer(status, route, (), (), (), (), (), "Unable to select evidence.", errors)

    async def _investigate(self, query, bubble, config, use_agents, route):
        history: list[Observation] = []
        collected: dict[str, Evidence] = {}
        status = "budget_exhausted"
        try:

            async with asyncio.timeout(self.limits.agent_seconds):
                if not use_agents:
                    call = ToolCall("search", {"query": query.text})
                    await self._observe(query, bubble, config, call, history, collected, route.bubbles)
                    if (route.names_specific_item and route.item_id and collected
                            and {e.item_id for e in collected.values()} == {route.item_id}):
                        await self._observe(query, bubble, config,
                                            ToolCall("read_item", {"item_id": route.item_id, "parts": [1, 2]}),
                                            history, collected, route.bubbles)
                    status = "failed" if any(h.error for h in history) else "direct"
                else:
                    for _ in range(self.limits.tool_calls):
                        action = await self.model.step(query, bubble, config.mode, tuple(history))
                        if isinstance(action, Finish):
                            status = ("partial" if collected else "failed") if any(h.error for h in history) else "complete"
                            break
                        if not isinstance(action, ToolCall):
                            history.append(Observation("invalid", error="invalid_action_schema"))
                            continue
                        await self._observe(query, bubble, config, action, history, collected, route.bubbles)
        except TimeoutError:
            status = "timeout"
            if history and history[-1].error == "in_progress":
                history[-1] = Observation(history[-1].tool, error="deadline_exceeded")
        except Exception as error:

            history.append(Observation("controller", error=type(error).__name__))
            status = "failed"
        return Investigation(bubble.id, status, tuple(collected.values()), tuple(history))

    async def _observe(self, query, bubble, config, call, history, collected, allowed_bubbles):
        position = len(history)
        history.append(Observation(call.name, error="in_progress"))
        try:
            self._validate_call(call, config, collected)
            records = await self.backend.execute(bubble.id, config.mode, call, query.scope, query.known_at,
                                                 allowed_bubbles=allowed_bubbles)
            if len(records) > 500:
                raise ValueError("backend_result_limit")
            if call.name == "read_item":
                if len(records) > len(call.args["parts"]) or any(len(e.text) > self.limits.part_chars for e in records):
                    raise ValueError("original_text_limit")
            limit = 10 if config.mode == "classic" else 20
            if call.name == "read_item":
                limit = len(call.args["parts"])
            accepted = []
            for e in records:
                if not isinstance(e, Evidence):
                    raise ValueError("invalid_evidence_schema")
                permitted = allowed_bubbles if config.mode == "own" else (bubble.id,)
                if e.bubble_id not in permitted or not e.visible(query.scope, query.known_at):
                    continue
                if call.name == "facts" and not e.visible(TimeScope(call.args["from"], call.args["to"]), query.known_at):
                    continue
                if call.name == "read_item" and e.item_id != call.args["item_id"]:
                    continue
                if e.unit_id in collected and collected[e.unit_id] != e:
                    raise ValueError("inconsistent_unit_identity")
                if e.unit_id not in {x.unit_id for x in accepted}:
                    accepted.append(e)
                if len(accepted) >= limit:
                    break
            for e in accepted:
                collected[e.unit_id] = e
            history[position] = Observation(call.name, tuple(accepted))
        except Exception as error:

            history[position] = Observation(call.name, error=str(error) if isinstance(error, ValueError) else type(error).__name__)

    def _validate_call(self, call, config, collected):
        allowed = {"search", "read_item"}
        if config.mode != "classic":
            allowed |= {"neighbors", "facts"}
        if call.name not in allowed or not isinstance(call.args, dict):
            raise ValueError("tool_not_allowed")
        schemas = {"search": {"query"}, "read_item": {"item_id", "parts"},
                   "neighbors": {"entity_id", "depth"}, "facts": {"entity_id", "from", "to"}}
        if set(call.args) != schemas[call.name]:
            raise ValueError("tool_argument_schema")
        args = call.args
        if call.name == "search":
            if not isinstance(args["query"], str) or not 1 <= len(args["query"].strip()) <= 2000:
                raise ValueError("invalid_search_query")
        elif call.name == "read_item":
            parts = args["parts"]
            if (not isinstance(parts, list) or not 1 <= len(parts) <= 3
                    or any(type(p) is not int or p < 1 for p in parts)
                    or len(set(parts)) != len(parts)):
                raise ValueError("invalid_parts")
            if args["item_id"] not in {e.item_id for e in collected.values()}:
                raise ValueError("item_not_discovered")
        else:
            known_entities = {x for e in collected.values() for x in e.global_entities}
            if args["entity_id"] not in known_entities:
                raise ValueError("entity_not_discovered")
            if call.name == "neighbors":
                if type(args["depth"]) is not int or not 1 <= args["depth"] <= 2:
                    raise ValueError("invalid_depth")
            else:
                TimeScope(args["from"], args["to"])

    def _context(self, investigations):
        seen, result, chars = set(), [], 0

        for row in zip_longest(*(i.evidence for i in investigations)):
            for e in row:
                if e is None or e.unit_id in seen:
                    continue
                seen.add(e.unit_id)
                if len(result) >= self.limits.max_context_units:
                    return tuple(result)
                if chars + len(e.text) <= self.limits.max_context_chars:
                    result.append(e)
                    chars += len(e.text)
        return tuple(result)
