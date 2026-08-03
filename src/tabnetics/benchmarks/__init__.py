"""Benchmark runner and profile surface backing ``tabnetics-benchmark`` / ``python -m tabnetics.benchmarks.cli``, exposing the profile registry for systematic paired comparisons and the runner that enforces the validation-catalog data policy: evidence-bearing runs use the HuggingFace mirror of public upstream sources and do not silently fall back to synthetic proxies."""

from .profiles import BEYONDARENA_MODEL_PARITY, BEYONDARENA_PARITY_BACKENDS, FS_METHOD_SETS
from .beyondarena_compare import *  # noqa: F401,F403
from .beyondarena_compare import __all__ as _beyondarena_compare_all
from .beyondarena_local import *  # noqa: F401,F403
from .beyondarena_local import __all__ as _beyondarena_local_all
from .beyondarena_materialize import *  # noqa: F401,F403
from .beyondarena_materialize import __all__ as _beyondarena_materialize_all
from .beyondarena_plan import *  # noqa: F401,F403
from .beyondarena_plan import __all__ as _beyondarena_plan_all
from .runner import *  # noqa: F401,F403

_AMBIGUOUS_CLI_EXPORTS = {"main"}

__all__ = sorted(
    {
        "BEYONDARENA_MODEL_PARITY",
        "BEYONDARENA_PARITY_BACKENDS",
        "FS_METHOD_SETS",
        *[name for name in _beyondarena_compare_all if name not in _AMBIGUOUS_CLI_EXPORTS],
        *[name for name in _beyondarena_local_all if name not in _AMBIGUOUS_CLI_EXPORTS],
        *[name for name in _beyondarena_materialize_all if name not in _AMBIGUOUS_CLI_EXPORTS],
        *[name for name in _beyondarena_plan_all if name not in _AMBIGUOUS_CLI_EXPORTS],
    }
)
