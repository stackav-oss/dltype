# content of conftest.py

from __future__ import annotations

import pytest

empty_mark = pytest.Mark("", args=(), kwargs={})


def _by_slow_marker(item: pytest.Item) -> bool:
    return item.get_closest_marker("slow", default=empty_mark) != empty_mark


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    items.sort(key=_by_slow_marker, reverse=False)
