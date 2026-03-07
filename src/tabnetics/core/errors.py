"""Stable exception types for validation/dataset integrity.

These are split out from `tabnetics.validation.suite` so that unit tests that
reload `tabnetics.validation.suite` (to simulate optional deps missing) do
not accidentally create a second set of exception classes. Importing these from
this module keeps identity stable across reloads.
"""


class DatasetIntegritySkipError(RuntimeError):
    """Raised when dataset integrity policy requests deterministic skip."""


class DatasetIntegrityPolicyError(RuntimeError):
    """Raised when dataset integrity policy requests hard failure."""

