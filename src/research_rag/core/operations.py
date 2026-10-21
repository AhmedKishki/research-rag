"""Every service operation a surface forwards, declared once.

An entry names an operation's arguments, their kinds and bounds, and the
`ResearchService` call it makes. The control API, the command line, the
workspace adapter, and the agent's search call `run`, so an argument is checked
in one place and no surface accepts what another refuses. The command line
reaches some operations and arguments no other surface does: in process, or
through the control API, which answers it alone.

A query string carries every value as text, so a number, a flag, or a comma
list written as text is read as its kind.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..project.policy import ResearchError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .service import ResearchService

# The passages one search may return, and the neighbours one passage read may
# carry. The agent tools state the same bounds in their schemas.
MAXIMUM_TOP_K = 50
MAXIMUM_CONTEXT_CHUNKS = 5


@dataclass(frozen=True, slots=True)
class Argument:
    """One named argument: its kind, whether it must be given, and its bounds.

    `kind` is `str`, `bool`, `int`, `float` (a positive number), `list` (of
    non-empty strings), or `dict`. A required `str` must not be blank, and a
    required `dict` with `minimum=1` must name something. A `command_line`
    argument is accepted only from the command line, and reaches the service only
    when that caller gives it.
    """

    name: str
    kind: type
    required: bool = False
    default: Any = None
    minimum: int | None = None
    maximum: int | None = None
    command_line: bool = False

    def read(self, value: Any) -> Any:
        if value is None:
            if self.required:
                raise ResearchError(f"{self.name} is required")
            return self.default
        reader = _READERS[self.kind]
        return reader(self, value)


def _refuse(argument: Argument, what: str) -> ResearchError:
    return ResearchError(f"{argument.name} must be {what}")


def _read_str(argument: Argument, value: Any) -> str:
    if not isinstance(value, str):
        raise _refuse(argument, "a string")
    if argument.required and not value.strip():
        raise _refuse(argument, "a non-empty string")
    return value


def _read_bool(argument: Argument, value: Any) -> bool:
    if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
        return value.strip().casefold() == "true"
    if not isinstance(value, bool):
        raise _refuse(argument, "true or false")
    return value


def _read_int(argument: Argument, value: Any) -> int:
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _refuse(argument, "an integer")
    low, high = argument.minimum, argument.maximum
    if (low is not None and value < low) or (high is not None and value > high):
        raise _refuse(argument, f"between {low} and {high}")
    return value


def _read_float(argument: Argument, value: Any) -> float:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise _refuse(argument, "a number") from None
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise _refuse(argument, "a positive number")
    return float(value)


def _read_list(argument: Argument, value: Any) -> list[str] | None:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()] or None
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise _refuse(argument, "a list of non-empty strings")
    return [item.strip() for item in value]


def _read_dict(argument: Argument, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _refuse(argument, "an object")
    if argument.minimum and len(value) < argument.minimum:
        raise _refuse(argument, "an object naming at least one key")
    return dict(value)


_READERS: dict[type, Callable[[Argument, Any], Any]] = {
    str: _read_str,
    bool: _read_bool,
    int: _read_int,
    float: _read_float,
    list: _read_list,
    dict: _read_dict,
}


Call = Callable[["ResearchService", dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class Operation:
    arguments: tuple[Argument, ...]
    call: Call
    command_line: bool = False


def _texts(*names: str) -> tuple[Argument, ...]:
    return tuple(Argument(name, list) for name in names)


_SEARCH_FILTERS = (
    "categories_any",
    "projects_any",
    "keywords",
    "languages_any",
    "authors_any",
    "titles_any",
    "source_ids",
    "exclude_source_ids",
)


async def _generations(service: ResearchService, _: dict[str, Any]) -> dict[str, Any]:
    # The inventory is already in the status payload, so a listing reads it
    # rather than walking the directory a second time with a second shape.
    status = await service.status()
    generations = status.get("generations")
    return {
        "generations": generations if isinstance(generations, list) else [],
        "retained_generation_count": int(status.get("retained_generation_count") or 0),
        "retained_generation_bytes": int(status.get("retained_generation_bytes") or 0),
        "current_generation_id": status.get("generation_id"),
    }


OPERATIONS: dict[str, Operation] = {
    "status": Operation((), lambda service, _: service.status()),
    # An agent and the workspace build every source; the command line may name
    # the ones to read, and may take sources out of the generation.
    "ingest": Operation(
        (
            Argument("force_recompute", bool, default=False),
            Argument("sources", list, command_line=True),
        ),
        lambda service, a: service.ingest(**a),
    ),
    "uningest": Operation(
        (Argument("sources", list, required=True),),
        lambda service, a: service.uningest(**a),
        command_line=True,
    ),
    # The engine chooses the method the generation can serve and always
    # reranks. Only the command line may name either, to reproduce a
    # measurement.
    "search": Operation(
        (
            Argument("query", str, required=True),
            Argument("top_k", int, default=10, minimum=1, maximum=MAXIMUM_TOP_K),
            *_texts(*_SEARCH_FILTERS),
            Argument("retrieval_method", str, command_line=True),
            Argument("rerank", bool, command_line=True),
        ),
        lambda service, a: service.search(
            a.pop("query"), **{"rerank": True, **a}, include_staleness=True
        ),
    ),
    # DeepSeek chooses each query; the reader's filters bound every one.
    "answer": Operation(
        (Argument("question", str, required=True), *_texts(*_SEARCH_FILTERS)),
        lambda service, a: service.answer(a.pop("question"), **a),
    ),
    "credentials_read": Operation((), lambda service, _: service.credentials_read()),
    "credentials_write": Operation(
        (Argument("api_key", str), Argument("clear", bool, default=False)),
        lambda service, a: service.credentials_write(**a),
    ),
    "list_sources": Operation(
        _texts("categories", "categories_any", "projects", "projects_any", "keywords"),
        lambda service, a: service.list_sources(**a),
    ),
    "get_passage": Operation(
        (
            Argument("chunk_id", str, required=True),
            Argument(
                "context_chunks",
                int,
                default=1,
                minimum=0,
                maximum=MAXIMUM_CONTEXT_CHUNKS,
            ),
        ),
        lambda service, a: service.get_passage(a.pop("chunk_id"), **a),
    ),
    "source_chunks": Operation(
        (
            Argument("source_id", str),
            Argument("source_path", str),
            Argument("page", int, default=1, minimum=1, maximum=1_000_000),
            Argument("page_size", int, default=20, minimum=1, maximum=50),
        ),
        lambda service, a: service.source_chunks(**a),
    ),
    "set_source_inclusion": Operation(
        (
            Argument("source_path", str),
            Argument("source_id", str),
            Argument("included", bool, required=True),
            Argument("reason", str),
        ),
        lambda service, a: service.set_source_inclusion(**a),
    ),
    "set_chunk_inclusion": Operation(
        (
            Argument("chunk_id", str, required=True),
            Argument("included", bool, required=True),
            Argument("reason", str),
        ),
        lambda service, a: service.set_chunk_inclusion(**a),
    ),
    "list_chunk_exclusions": Operation(
        (), lambda service, _: service.list_chunk_exclusions()
    ),
    "set_source_metadata": Operation(
        (
            Argument("metadata", dict, required=True),
            Argument("source_path", str),
            Argument("source_id", str),
        ),
        lambda service, a: service.set_source_metadata(**a),
    ),
    "list_generations": Operation((), _generations),
    "use_generation": Operation(
        (Argument("generation_id", str, required=True),),
        lambda service, a: service.use_generation(**a),
    ),
    # The confirmation is carried, never defaulted: the service refuses one that
    # does not repeat the id, so a click cannot mean yes.
    "remove_generation": Operation(
        (
            Argument("generation_id", str, required=True),
            Argument("confirm", str, required=True),
        ),
        lambda service, a: service.remove_generation(**a),
    ),
    "settings_read": Operation((), lambda service, _: service.settings_read()),
    # The revision is required: a write naming none could land on a change made
    # since the page read the settings.
    "settings_write": Operation(
        (
            Argument("values", dict, required=True, minimum=1),
            Argument("expected_revision", str, required=True),
            Argument("confirm", bool, default=False),
        ),
        lambda service, a: service.settings_write(a.pop("values"), **a),
    ),
    "search_stats": Operation(
        (
            Argument("days", float),
            Argument("top", int, default=20, minimum=1, maximum=100),
            Argument("largest_by", str, default="passages"),
        ),
        lambda service, a: service.search_stats(
            since_days=a["days"], top=a["top"], largest_by=a["largest_by"]
        ),
    ),
    "search_history": Operation(
        (
            Argument("limit", int, default=20, minimum=1, maximum=200),
            Argument("days", float),
        ),
        lambda service, a: service.search_history(
            limit=a["limit"], since_days=a["days"]
        ),
    ),
    "clear_search_history": Operation(
        (), lambda service, _: service.clear_search_history()
    ),
}


def arguments(
    name: str, payload: Mapping[str, Any], *, command_line: bool = False
) -> dict[str, Any]:
    """Every argument of `name`, read from `payload` and checked, or a refusal."""

    operation = OPERATIONS.get(name)
    if operation is None:
        raise ResearchError(f"Unknown operation: {name}")
    if operation.command_line and not command_line:
        raise ResearchError(f"{name} is available from the command line only")
    accepted = {
        argument.name: argument
        for argument in operation.arguments
        if command_line or not argument.command_line
    }
    unknown = sorted(set(payload) - set(accepted))
    if unknown:
        raise ResearchError(f"{name} does not take: {', '.join(unknown)}")
    return {
        argument.name: argument.read(payload.get(argument.name))
        for argument in accepted.values()
        if not argument.command_line or argument.name in payload
    }


async def run(
    service: ResearchService,
    name: str,
    payload: Mapping[str, Any],
    *,
    command_line: bool = False,
) -> dict[str, Any]:
    """Check `payload` against operation `name` and make its service call."""

    checked = arguments(name, payload, command_line=command_line)
    return await OPERATIONS[name].call(service, checked)
