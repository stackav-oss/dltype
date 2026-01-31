"""Constants related to the dltype library."""

import typing
import warnings

from pydantic_settings import BaseSettings, SettingsConfigDict


class _Env(BaseSettings):
    """Environment variables controlling dltype behavior."""

    model_config = SettingsConfigDict(
        frozen=True,
        env_prefix="DLTYPE_",
        case_sensitive=False,
    )

    DISABLE: bool = False
    """Disable all dltype checking."""

    DEBUG_MODE: bool = False
    """If true, set debug mode enabled for debugging library issues."""


# Constants
__env = _Env()
PYDANTIC_INFO_KEY: typing.Final = "__dltype__"
DEBUG_MODE: typing.Final = __env.DEBUG_MODE
MAX_ACCEPTABLE_EVALUATION_TIME_NS: typing.Final = int(5e9)  # 5ms
GLOBAL_DISABLE: typing.Final = __env.DISABLE


if GLOBAL_DISABLE:
    warnings.warn(
        "DLType disabled via environment variable, all decorated functions not explicitly marked with enable=True will be turned off.",
        UserWarning,
        stacklevel=1,
    )
