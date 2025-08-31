"""Constants related to the dltype library."""

import typing

# Constants
PYDANTIC_INFO_KEY: typing.Final = "__dltype__"
DEBUG_MODE: typing.Final = False
MAX_ACCEPTABLE_EVALUATION_TIME_NS: typing.Final = int(5e9)  # 5ms
