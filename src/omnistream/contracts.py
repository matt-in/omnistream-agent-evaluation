"""Shared interfaces. Adapters must not receive evaluation answers or qrels."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol


Mode = Literal["external", "classic", "graphiti", "own"]


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Timestamps must include a UTC offset")
    return result


@dataclass(frozen=True)
class Configuration:
    name: str
    mode: Mode
    subagents: bool = False

    def __post_init__(self):
        if self.mode not in {"external", "classic", "graphiti", "own"}:
            raise ValueError("Unknown retrieval mode")
        if self.mode == "external" and self.subagents:
            raise ValueError("The external baseline is single shot")


CONFIGURATIONS = (
    Configuration("external", "external"),
    Configuration("classic", "classic"),
    Configuration("graphiti", "graphiti"),
    Configuration("own", "own"),
    Configuration("own_agents", "own", True),
    Configuration("classic_agents", "classic", True),
)


@dataclass(frozen=True)
class Bubble:
    id: str
    description: str
    auto_summary: str = ""


@dataclass(frozen=True)
class TimeScope:
    """Point at start when end is absent; otherwise a half-open [start,end) range."""

    start: str
    end: str | None = None

    def __post_init__(self):
        start = timestamp(self.start)
        if self.end is not None and timestamp(self.end) <= start:
            raise ValueError("Time range must be nonempty")


@dataclass(frozen=True)
class Query:
    text: str
    known_at: str
    scope: TimeScope | None = None
    pin: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    def __post_init__(self):
        timestamp(self.known_at)
        if not self.text.strip():
            raise ValueError("Question is empty")


@dataclass(frozen=True)
class Route:
    bubbles: tuple[str, ...]
    strategy: Literal["simple", "multi_hop"]
    scope: TimeScope
    names_specific_item: bool = False
    item_id: str | None = None
    rationale: str = ""


@dataclass(frozen=True)
class Evidence:
    unit_id: str
    item_id: str
    bubble_id: str
    text: str
    start: int
    end: int
    published_at: str
    available_at: str
    valid_from: str | None = None
    valid_to: str | None = None
    invalidated_at: str | None = None
    grounded: bool = True
    credibility: float = 0.5
    subject: str | None = None
    relation: str | None = None
    object: str | None = None
    global_entities: tuple[str, ...] = ()
    kind: Literal["fact", "chunk", "original"] = "fact"
    path: tuple[str, ...] = ()
    source_url: str = ""

    def __post_init__(self):
        if not all((self.unit_id, self.item_id, self.bubble_id, self.text)):
            raise ValueError("Evidence needs identity, provenance and text")
        if type(self.grounded) is not bool:
            raise ValueError("grounded must be a boolean")
        if self.start < 0 or self.end != self.start + len(self.text):
            raise ValueError("Evidence offsets are Unicode character offsets in clean text")
        if not 0 <= self.credibility <= 1:
            raise ValueError("Credibility must be between zero and one")
        timestamp(self.published_at)
        timestamp(self.available_at)
        if self.valid_from:
            timestamp(self.valid_from)
        if self.valid_to:
            timestamp(self.valid_to)
        if self.invalidated_at:
            timestamp(self.invalidated_at)
        if self.valid_from and self.valid_to:
            if timestamp(self.valid_to) <= timestamp(self.valid_from):
                raise ValueError("Invalid fact validity interval")

    def visible(self, scope: TimeScope, known_at: str) -> bool:
        known = timestamp(known_at)
        if timestamp(self.published_at) > known or timestamp(self.available_at) > known:
            return False
        if not self.grounded:
            return False
        start = timestamp(self.valid_from) if self.valid_from else None
        invalidation_known = self.invalidated_at is None or timestamp(self.invalidated_at) <= known
        end = timestamp(self.valid_to) if self.valid_to and invalidation_known else None
        qstart = timestamp(scope.start)
        if scope.end is None:
            return (start is None or start <= qstart) and (end is None or qstart < end)
        return (start is None or start < timestamp(scope.end)) and (end is None or qstart < end)


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Finish:
    summary: str = ""


@dataclass(frozen=True)
class Observation:
    tool: str
    evidence: tuple[Evidence, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class Claim:
    text: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class Draft:
    claims: tuple[Claim, ...]
    abstain: bool = False


class Model(Protocol):
    """Production adapter maps these typed calls to validated structured LLM output."""

    async def route(self, query: Query, catalog: tuple[Bubble, ...]) -> Route: ...

    async def step(
        self, query: Query, bubble: Bubble, mode: Mode,
        history: tuple[Observation, ...],
    ) -> ToolCall | Finish: ...

    async def synthesize(
        self, query: Query, evidence: tuple[Evidence, ...],
        conflicts: tuple[dict, ...],
    ) -> Draft: ...


class Backend(Protocol):
    """Apply bubble, knowledge cutoff and time filters BEFORE ranking, then limit.

    The controller also validates returned records. Evidence spans must come
    from clean stored text; graph fact paraphrases need a supporting source span.
    read_item returns at most one Evidence per requested part. No raw Cypher is
    passed through this boundary. Global subject/object IDs must be canonical.
    """

    async def execute(
        self, bubble_id: str, mode: Mode, call: ToolCall,
        scope: TimeScope, known_at: str, *, allowed_bubbles: tuple[str, ...],
    ) -> tuple[Evidence, ...]: ...
