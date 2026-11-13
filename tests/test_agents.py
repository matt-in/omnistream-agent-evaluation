import asyncio
import unittest
from dataclasses import replace

from omnistream.agents import AgentRunner, Limits, merge_conflicts
from omnistream.contracts import (
    Bubble, Claim, Configuration, Draft, Evidence, Finish, Query, Route, TimeScope, ToolCall,
)


NOW = "2026-11-10T12:00:00Z"
BEFORE = "2026-11-01T00:00:00Z"
SCOPE = TimeScope(NOW)
CATALOG = (Bubble("news", "News"), Bubble("finance", "Finance"))
OWN = Configuration("own_agents", "own", True)


def fact(unit="n1", bubble="news", text="Oriole is based in Meridian.", **kw):
    return Evidence(unit, "item_" + unit, bubble, text, 0, len(text), BEFORE, BEFORE,
                    global_entities=("oriole", "meridian"), **kw)


class Model:
    def __init__(self, route=None, actions=None, draft=None):
        self.plan = route or Route(("news",), "multi_hop", SCOPE)
        self.actions = actions
        self.draft = draft
        self.synthesis_calls = 0

    async def route(self, query, catalog):
        if isinstance(self.plan, Exception):
            raise self.plan
        return self.plan

    async def step(self, query, bubble, mode, history):
        if self.actions:
            return self.actions(history)
        return Finish() if history else ToolCall("search", {"query": query.text})

    async def synthesize(self, query, evidence, conflicts):
        self.synthesis_calls += 1
        return self.draft if self.draft is not None else Draft((Claim(evidence[0].text, (evidence[0].unit_id,)),))


class Backend:
    def __init__(self, records=None):
        self.records = records if records is not None else (fact(),)
        self.calls = []

    async def execute(self, bubble_id, mode, call, scope, known_at, *, allowed_bubbles=()):
        self.calls.append((bubble_id, mode, call, scope, known_at))
        return self.records


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def run_agent(self, model=None, backend=None, config=OWN, query=None, limits=None):
        return await AgentRunner(model or Model(), backend or Backend(), CATALOG, limits).run(
            query or Query("Where is Oriole?", NOW, SCOPE), config)

    async def test_parallel_bubbles_use_concurrent_tasks(self):
        class BarrierBackend(Backend):
            def __init__(self):
                super().__init__()
                self.arrived = set()
                self.event = asyncio.Event()

            async def execute(self, bubble_id, mode, call, scope, known_at, **kwargs):
                self.arrived.add(bubble_id)
                if len(self.arrived) == 2:
                    self.event.set()
                await self.event.wait()
                return (fact(bubble_id, bubble_id),)
        backend = BarrierBackend()
        result = await self.run_agent(Model(Route(("news", "finance"), "multi_hop", SCOPE)), backend,
                                      limits=Limits(agent_seconds=.5))
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.investigations), 2)
        self.assertEqual({e.bubble_id for e in result.evidence}, {"news", "finance"})

    async def test_unknown_route_fails_before_backend(self):
        backend = Backend()
        answer = await self.run_agent(Model(Route(("missing",), "simple", SCOPE)), backend)
        self.assertEqual(answer.status, "routing_failed")
        self.assertFalse(backend.calls)

    async def test_provider_routing_failure_is_recorded(self):
        answer = await self.run_agent(Model(RuntimeError("provider unavailable")))
        self.assertEqual(answer.status, "routing_failed")
        self.assertEqual(answer.errors, ("RuntimeError",))

    async def test_synthesis_connection_failure_preserves_evidence(self):
        class FailingModel(Model):
            async def synthesize(self, query, evidence, conflicts):
                raise ConnectionError("Endpoint unavailable")

        answer = await self.run_agent(FailingModel())
        self.assertEqual(answer.status, "synthesis_failed")
        self.assertEqual(answer.errors, ("ConnectionError",))
        self.assertEqual([e.unit_id for e in answer.evidence], ["n1"])
        self.assertFalse(answer.claims)

    async def test_pins_and_exclusions_override_router(self):
        backend = Backend((fact("f1", "finance"), fact("n1")))
        query = Query("Where?", NOW, SCOPE, pin=("finance",), exclude=("news",))
        answer = await self.run_agent(backend=backend, query=query)
        self.assertEqual(answer.route.bubbles, ("finance",))
        self.assertEqual([e.bubble_id for e in answer.evidence], ["finance"])

    async def test_conflicting_pin_and_exclusion_rejected(self):
        with self.assertRaises(ValueError):
            await self.run_agent(query=Query("Where?", NOW, SCOPE, ("news",), ("news",)))

    async def test_empty_selection_does_not_search_all_namespaces(self):
        backend = Backend()
        result = await self.run_agent(backend=backend, query=Query("Where?", NOW, SCOPE, exclude=("news",)))
        self.assertEqual(result.status, "no_bubbles")
        self.assertFalse(backend.calls)

    async def test_classic_denies_graph_tools_and_bounds_invalid_attempts(self):
        backend = Backend()
        model = Model(actions=lambda h: ToolCall("neighbors", {"entity_id": "oriole", "depth": 1}))
        result = await self.run_agent(model, backend, Configuration("classic_agents", "classic", True))
        self.assertEqual(len(result.investigations[0].history), 6)
        self.assertTrue(all(h.error == "tool_not_allowed" for h in result.investigations[0].history))
        self.assertFalse(backend.calls)

    async def test_depth_cannot_exceed_two(self):
        actions = lambda h: (ToolCall("search", {"query": "Oriole"}) if not h else
                             ToolCall("neighbors", {"entity_id": "oriole", "depth": 3}) if len(h) == 1 else Finish())
        backend = Backend()
        result = await self.run_agent(Model(actions=actions), backend)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(result.investigations[0].history[1].error, "invalid_depth")

    async def test_graph_tool_requires_discovered_entity(self):
        result = await self.run_agent(Model(actions=lambda h: ToolCall("neighbors", {"entity_id": "invented", "depth": 2})))
        self.assertEqual(result.investigations[0].history[0].error, "entity_not_discovered")

    async def test_original_requires_discovered_item_and_three_parts_maximum(self):
        for args, expected in [({"item_id": "private", "parts": [1]}, "item_not_discovered"),
                               ({"item_id": "item_n1", "parts": [1, 2, 3, 4]}, "invalid_parts")]:
            with self.subTest(args=args):
                result = await self.run_agent(Model(actions=lambda h: ToolCall("read_item", args)))
                self.assertEqual(result.investigations[0].history[0].error, expected)

    async def test_six_calls_are_a_hard_limit(self):
        backend = Backend()
        result = await self.run_agent(Model(actions=lambda h: ToolCall("search", {"query": "Oriole"})), backend)
        self.assertEqual(len(backend.calls), 6)
        self.assertEqual(result.investigations[0].status, "budget_exhausted")
        self.assertEqual(result.status, "partial")

    async def test_deadline_includes_model_wait(self):
        class SlowModel(Model):
            async def step(self, *args):
                await asyncio.sleep(1)
                return Finish()
        answer = await self.run_agent(SlowModel(), limits=Limits(agent_seconds=.01))
        self.assertEqual(answer.investigations[0].status, "timeout")

    async def test_partial_evidence_survives_backend_timeout(self):
        class SlowBackend(Backend):
            async def execute(self, *args, **kwargs):
                if self.calls:
                    await asyncio.sleep(1)
                return await super().execute(*args, **kwargs)
        answer = await self.run_agent(Model(actions=lambda h: ToolCall("search", {"query": "Oriole"})),
                                      SlowBackend(), limits=Limits(agent_seconds=.02))
        self.assertEqual(answer.status, "partial")
        self.assertEqual(len(answer.evidence), 1)
        self.assertEqual(answer.investigations[0].history[-1].error, "deadline_exceeded")
        self.assertEqual(len(answer.investigations[0].history), 2)

    async def test_unseen_citation_is_rejected(self):
        answer = await self.run_agent(Model(draft=Draft((Claim("A claim", ("invented",)),))))
        self.assertEqual(answer.status, "synthesis_failed")
        self.assertFalse(answer.claims)

    async def test_uncited_claim_is_rejected(self):
        answer = await self.run_agent(Model(draft=Draft((Claim("A claim", ()),))))
        self.assertEqual(answer.status, "synthesis_failed")

    async def test_citation_validation_does_not_claim_semantic_entailment(self):
        answer = await self.run_agent(Model(draft=Draft((Claim("Oriole is on Mars.", ("n1",)),))))
        self.assertEqual(answer.status, "ok")
        self.assertEqual(answer.citations[0]["item_id"], "item_n1")


    async def test_no_evidence_skips_synthesis(self):
        model = Model()
        answer = await self.run_agent(model, Backend(()))
        self.assertEqual(answer.status, "no_evidence")
        self.assertEqual(model.synthesis_calls, 0)

    async def test_backend_records_are_checked_for_scope_time_and_grounding(self):
        good = fact()
        records = (replace(good, unit_id="wrong", bubble_id="finance"),
                   replace(good, unit_id="ungr", grounded=False),
                   replace(good, unit_id="future", available_at="2026-11-11T00:00:00Z"),
                   replace(good, unit_id="expired", valid_to=BEFORE), good)
        answer = await self.run_agent(backend=Backend(records))
        self.assertEqual([e.unit_id for e in answer.evidence], ["n1"])

    async def test_named_original_is_read_only_after_matching_search(self):
        route = Route(("news",), "simple", SCOPE, True, "item_n1")
        backend = Backend()
        answer = await self.run_agent(Model(route), backend)
        self.assertEqual([c[2].name for c in backend.calls], ["search", "read_item"])
        self.assertEqual(answer.status, "ok")

    async def test_context_limits_preserve_bubble_balance(self):
        class ScopedBackend(Backend):
            async def execute(self, bubble_id, *args, **kwargs):
                return tuple(fact(bubble_id + str(i), bubble_id) for i in range(4))
        answer = await self.run_agent(Model(Route(("news", "finance"), "multi_hop", SCOPE)),
                                      ScopedBackend(), limits=Limits(max_context_units=2))
        self.assertEqual(len(answer.evidence), 2)
        self.assertEqual({e.bubble_id for e in answer.evidence}, {"news", "finance"})

    async def test_caller_time_scope_overrides_model(self):
        backend = Backend()
        await self.run_agent(Model(Route(("news",), "simple", TimeScope(BEFORE))), backend)
        self.assertEqual(backend.calls[0][3], SCOPE)

    async def test_model_time_scope_used_when_caller_omits_it(self):
        backend = Backend()
        await self.run_agent(Model(Route(("news",), "simple", TimeScope(BEFORE))), backend,
                             query=Query("Where?", NOW))
        self.assertEqual(backend.calls[0][3], TimeScope(BEFORE))

    async def test_one_bubble_failure_does_not_erase_other_results(self):
        class FailingBackend(Backend):
            async def execute(self, bubble_id, *args, **kwargs):
                if bubble_id == "finance":
                    raise ConnectionError("unavailable")
                return (fact(),)
        answer = await self.run_agent(Model(Route(("news", "finance"), "multi_hop", SCOPE)), FailingBackend())
        self.assertTrue(answer.evidence)
        self.assertEqual(answer.investigations[1].history[0].error, "ConnectionError")

    async def test_external_requires_dedicated_adapter(self):
        with self.assertRaises(ValueError):
            await self.run_agent(config=Configuration("external", "external"))

    async def test_own_paths_may_cross_only_explicitly_selected_bubbles(self):
        records = (fact("news1"), fact("finance1", "finance"), fact("research1", "research"))
        answer = await self.run_agent(Model(Route(("news", "finance"), "multi_hop", SCOPE)), Backend(records))
        self.assertEqual({e.bubble_id for e in answer.evidence}, {"news", "finance"})

    async def test_facts_time_window_cannot_be_ignored_by_backend(self):
        old = fact(valid_from=BEFORE, valid_to="2026-11-05T00:00:00Z")
        now = fact("new", valid_from="2026-11-05T00:00:00Z")
        scope = TimeScope(BEFORE, "2026-11-10T00:00:00Z")
        actions = lambda h: (ToolCall("search", {"query": "Oriole"}) if not h else
                             ToolCall("facts", {"entity_id": "oriole", "from": "2026-11-06T00:00:00Z", "to": "2026-11-07T00:00:00Z"}) if len(h) == 1 else Finish())
        answer = await self.run_agent(Model(actions=actions), Backend((old, now)), query=Query("History?", NOW, scope))
        self.assertEqual([e.unit_id for e in answer.investigations[0].history[1].evidence], ["new"])


class TemporalAndMergeTests(unittest.TestCase):
    def test_later_invalidation_does_not_leak_into_earlier_knowledge_state(self):
        old = fact(valid_from=BEFORE, valid_to="2026-11-05T00:00:00Z",
                   invalidated_at="2026-11-05T00:01:00Z")
        future_scope = TimeScope("2026-11-06T00:00:00Z")
        self.assertTrue(old.visible(future_scope, "2026-11-04T00:00:00Z"))
        self.assertFalse(old.visible(future_scope, NOW))

    def test_validity_end_is_exclusive_and_knowledge_cutoff_is_separate(self):
        old = fact(valid_from=BEFORE, valid_to="2026-11-05T00:00:00Z")
        self.assertFalse(old.visible(TimeScope("2026-11-05T00:00:00Z"), NOW))
        late = replace(old, available_at="2026-11-06T00:00:00Z")
        self.assertFalse(late.visible(TimeScope("2026-11-04T00:00:00Z"), "2026-11-04T12:00:00Z"))
        self.assertTrue(late.visible(TimeScope("2026-11-04T00:00:00Z"), NOW))

    def test_invalid_offsets_and_naive_dates_are_rejected(self):
        with self.assertRaises(ValueError):
            replace(fact(), end=2)
        with self.assertRaises(ValueError):
            TimeScope("2026-11-10T12:00:00")

    def test_conflicts_keep_both_sides_and_follow_credibility_threshold(self):
        a = fact(subject="oriole", relation="ceo", object="niko", credibility=.8)
        b = replace(a, unit_id="b", object="leda", published_at="2026-11-03T00:00:00Z", credibility=.6)
        c = replace(a, unit_id="c", object="other", published_at="2026-11-04T00:00:00Z", credibility=.59)
        result = merge_conflicts((a, b, c))[0]
        self.assertEqual(set(result["evidence_ids"]), {"n1", "b", "c"})
        self.assertEqual(result["preferred"], "b")

    def test_equal_timestamp_disagreement_has_no_arbitrary_preference(self):
        a = fact(subject="oriole", relation="ceo", object="niko", credibility=.8)
        b = replace(a, unit_id="b", object="leda")
        self.assertIsNone(merge_conflicts((a, b))[0]["preferred"])

    def test_successive_ceos_are_not_a_simultaneous_conflict(self):
        a = fact(subject="oriole", relation="ceo", object="niko", valid_to="2026-11-05T00:00:00Z")
        b = replace(a, unit_id="b", object="leda", valid_from="2026-11-05T00:00:00Z", valid_to=None)
        self.assertEqual(merge_conflicts((a, b)), ())


class ContractTests(unittest.TestCase):
    def test_limits_reject_invalid_budgets_and_deadlines(self):
        invalid = (
            {"tool_calls": 2.5}, {"tool_calls": True}, {"max_context_units": 0},
            {"agent_seconds": float("nan")}, {"agent_seconds": float("inf")},
            {"router_seconds": -1}, {"synthesis_seconds": "60"},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Limits(**values)

    def test_grounding_flag_requires_a_boolean(self):
        for value in ("false", 0, 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(fact(), grounded=value)
        self.assertFalse(replace(fact(), grounded=False).visible(SCOPE, NOW))
