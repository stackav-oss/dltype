"""Utilities for logging in the dltype library."""

import logging
import typing

from dltype._lib import _constants


class DummyLogger:
    def debug(self, *args: typing.Any) -> None:  # noqa: ANN401
        pass

    def info(self, *args: typing.Any) -> None:  # noqa: ANN401
        pass

    def warning(self, *args: typing.Any) -> None:  # noqa: ANN401
        pass

    def error(self, *args: typing.Any) -> None:  # noqa: ANN401
        pass

    def exception(self, *args: typing.Any) -> None:  # noqa: ANN401
        pass


def get_logger(name: str) -> logging.Logger | DummyLogger:
    if _constants.GLOBAL_DISABLE:
        return DummyLogger()
    if not _constants.DEBUG_MODE:
        return DummyLogger()
    return logging.getLogger(name)
