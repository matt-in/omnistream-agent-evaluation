"""Scripted model and lexical retrieval adapters for development."""

import json
import re
from dataclasses import replace
from pathlib import Path

from .agents import AgentRunner
from .contracts import (
    Bubble, Claim, Configuration, Draft, Evidence, Finish, Route, TimeScope, ToolCall,
)


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


CATALOG = (
    Bubble("news", "News announcements events launch project Mira Vale Lumen relay sponsor"),
    Bubble("finance", "Finance company Oriole Labs CEO headquarters city sponsor grant funding"),
    Bubble("research", "Research Tern Institute director leads researcher paper algorithm designed developed"),
)


class FixtureBackend:
    """Lexical development store shared across retrieval modes."""

    def __init__(self, items_path):
        self.items = tuple(Evidence(**{**x, "global_entities": tuple(x.get("global_entities", ()))})
                           for x in (json.loads(line) for line in Path(items_path).read_text(encoding="utf-8").splitlines() if line))
        self.calls = []

    async def execute(self, bubble_id, mode, call, scope, known_at, *, allowed_bubbles):
        self.calls.append((bubble_id, mode, call.name))
        candidates = [e for e in self.items if e.bubble_id == bubble_id and e.visible(scope, known_at)]
        if call.name == "search":
            query = words(call.args["query"])
            candidates.sort(key=lambda e: (-len(query & words(e.text)), e.unit_id))
            return tuple(e for e in candidates if query & words(e.text))
        if call.name == "read_item":
            matching = [e for e in candidates if e.item_id == call.args["item_id"]]
            return tuple(replace(e, unit_id=e.item_id + ":part1", kind="original")
                         for e in matching[:1]) if 1 in call.args["parts"] else ()
        candidates = [e for e in candidates if call.args["entity_id"] in e.global_entities]
        if call.name == "facts":
            narrower = TimeScope(call.args["from"], call.args["to"])
            candidates = [e for e in candidates if e.visible(narrower, known_at)]
        return tuple(candidates)


class FixtureModel:
    """Scripted control decisions, lexical routing and extractive answers for tests."""

    async def route(self, query, catalog):
        text = words(query.text)
        scored = [(len(text & words(b.description)), b.id) for b in catalog]
        multi = bool(text & {"sponsor", "funded", "funds", "developed", "behind"})
        selected = tuple(b.id for b in catalog) if multi else (max(scored)[1],)
        return Route(selected, "multi_hop" if multi else "simple",
                     query.scope or TimeScope(query.known_at), rationale="fixture lexical router")

    async def step(self, query, bubble, mode, history):
        if not history:
            return ToolCall("search", {"query": query.text})
        if len(history) == 1 and history[0].evidence:
            return ToolCall("read_item", {"item_id": history[0].evidence[0].item_id, "parts": [1]})
        return Finish("fixture investigation complete")

    async def synthesize(self, query, evidence, conflicts):
        if not evidence:
            return Draft((), True)
        return Draft(tuple(Claim(e.text, (e.unit_id,)) for e in evidence[:4]))


class FixtureSystem:
    def __init__(self, items_path):
        self.backend = FixtureBackend(items_path)
        self.runner = AgentRunner(FixtureModel(), self.backend, CATALOG)

    async def answer(self, case, config):
        effective = Configuration(config.name, "classic" if config.mode == "external" else config.mode,
                                  config.subagents)
        return await self.runner.run(case.query, effective)
