"""Composition of the controller and project retrieval adapters."""

from .agents import AgentRunner, Answer, Limits
from .contracts import CONFIGURATIONS, Backend, Bubble, Configuration, Model, Query, SourceStore
from .evaluation import EvaluationCase, SystemAdapter


class RetrievalAdapters:
    def __init__(self, *, classic: Backend, graphiti: Backend, own: Backend):
        self.backends = {"classic": classic, "graphiti": graphiti, "own": own}

    async def execute(self, bubble_id, mode, call, scope, known_at, *, allowed_bubbles):
        return await self.backends[mode].execute(
            bubble_id, mode, call, scope, known_at, allowed_bubbles=allowed_bubbles,
        )


class ProjectSystem:
    """Custom graph retrieval for the application and six configurations for evaluation."""

    def __init__(self, model: Model, retrieval: RetrievalAdapters, catalog: tuple[Bubble, ...],
                 originals: SourceStore, external: SystemAdapter, limits: Limits | None = None):
        self.runner = AgentRunner(model, retrieval, catalog, limits, source_store=originals)
        self.external = external
        self.application = next(config for config in CONFIGURATIONS if config.name == "own_agents")

    async def ask(self, query: Query) -> Answer:
        return await self.runner.run(query, self.application)

    async def answer(self, case: EvaluationCase, config: Configuration) -> Answer:
        if config.mode == "external":
            return await self.external.answer(case, config)
        return await self.runner.run(case.query, config)
