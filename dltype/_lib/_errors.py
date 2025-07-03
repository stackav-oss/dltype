"""Errors for the dltype library."""


class DLTypeError(TypeError):
    """An error raised when a type assertion is hit."""


class DLTypeShapeError(DLTypeError):
    """An error raised when a shape assertion is hit."""


class DLTypeDtypeError(DLTypeError):
    """An error raised when a dtype assertion is hit."""


class DLTypeDuplicateError(DLTypeError):
    """An error raised when a duplicate tensor name is hit."""


class DLTypeInvalidReferenceError(DLTypeError):
    """An error raised when an invalid reference is hit."""


class DLTypeScopeProviderError(DLTypeError):
    """An error raised when an invalid scope provider is hit."""
