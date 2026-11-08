"""Checkpointer construction.

A checkpointer snapshots the graph's state after every node, keyed by a thread
id. Snapshots are serialized, and LangGraph only deserializes custom types you
have explicitly allowed (it warns today and will block in a future version), so
every type that can appear in `ReviewState` is listed here. A test fails if one
is forgotten.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

# (module, class name) of every custom type stored in ReviewState.
STATE_TYPES: list[tuple[str, str]] = [
    ("pr_agent.models", "ChangedFile"),
    ("pr_agent.models", "Finding"),
    ("pr_agent.models", "Severity"),
    ("pr_agent.models", "Category"),
    ("pr_agent.models", "RetrievedContext"),
    ("pr_agent.models", "HunkContext"),
    ("pr_agent.diff_parser", "DiffHunk"),
]


def checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES)


def memory_checkpointer() -> MemorySaver:
    """In RAM: right for tests and experiments, gone when the process exits."""
    return MemorySaver(serde=checkpoint_serializer())


@contextmanager
def sqlite_checkpointer(path: str | Path) -> Iterator[SqliteSaver]:
    """On disk: state survives a crash or restart. Use as a context manager."""
    # check_same_thread=False because LangGraph may touch the connection from worker threads.
    connection = sqlite3.connect(str(path), check_same_thread=False)
    try:
        yield SqliteSaver(connection, serde=checkpoint_serializer())
    finally:
        connection.close()
