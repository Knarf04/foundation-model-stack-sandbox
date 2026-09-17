"""
Shared argument checks.

The contract is the ValueError, not its wording -- these exist so the modules
below phrase the same two complaints the same way. range_spec deliberately does
NOT use them: it stays import-free for minimal_reference's sake.
"""

from typing import Sequence


def check_shape(name: str, x, expected: Sequence[int], note: str = "") -> None:
    """Raise unless x.shape == expected."""
    got = tuple(x.shape)
    want = tuple(expected)
    if got != want:
        suffix = f" ({note})" if note else ""
        raise ValueError(f"{name} shape {got} != {want}{suffix}")


def check_paired(name_a: str, a, name_b: str, b, note: str = "") -> None:
    """Raise unless a and b are both given or both None."""
    if (a is None) != (b is None):
        suffix = f" ({note})" if note else ""
        raise ValueError(
            f"{name_a} and {name_b} must both be provided or both be "
            f"None{suffix}"
        )
