"""Triton-side port of Module 4's canonical scoping mask (pure python).

This module mirrors ``ouroboros-core/src/scoping.py`` CASE-FOR-CASE so the
triton repo and the core repo agree on Paper Table 1: hierarchical attention
mask rules over tokens grouped into contiguous regions tagged GLOBAL or LOCAL.

Table 1 (M[i,j] as query-i-sees-key-j):
    M[i,j] = 1  if i is LOCAL and j is GLOBAL       function bodies see signatures
    M[i,j] = 1  if i == j                           self
    M[i,j] = 1  if i, j both LOCAL, same scope      coherent generation in block
    M[i,j] = 0  if i is GLOBAL and j is LOCAL       global KV-cache insulated
    M[i,j] = 0  if i, j LOCAL siblings, diff scopes no sibling bleed

Spec ambiguities resolved identically to core (documented deliberately):
  * GLOBAL x GLOBAL visibility is not listed in Table 1; it is set to 1 so
    signatures see each other and so the mask is symmetric exactly on
    {same-local-scope} U {global-global} pairs. Hence the mask is
    intentionally NOT symmetric overall: LOCAL-row/GLOBAL-col = 1 but the
    mirrored GLOBAL-row/LOCAL-col = 0.
  * Spans may be given unordered; they are accepted iff they form an exact
    partition of [0, L). Zero-length, negative, unknown-scope, gapped or
    overlapping regions raise ValueError. Empty spans list => L = 0.
  * ``global_rows_frozen`` is the assert-style True/False check "did global
    rows stay frozen under an [EXPAND]?"; it NEVER raises. (Core also ships a
    ``frozen_global_columns`` mask rebuilder; this port only needs the check,
    which the kernel-side tests use to validate post-expansion masks.)
  * Historical naming note kept from core: the helper freezes GLOBAL *rows*
    (a global token's output depends only on other globals, so its cached
    state survives local [EXPAND] mutations); the old name mentions columns.

The resulting LxL boolean mask feeds :func:`src.kernels.reference_attention.
golden_attention` as its ``scope_mask`` argument (truthy = allowed) and, per
token-per-block simplification there, one row per logical position.
"""

from bisect import bisect_right

__all__ = [
    "GLOBAL",
    "LOCAL",
    "Scope",
    "build_scope_tags",
    "same_local_scope",
    "block_sparse_mask",
    "global_rows_frozen",
    "frozen_global_rows",
]

GLOBAL = "global"
LOCAL = "local"


class Scope:
    """Enum-like scope constants (plain strings, per Module 4 spec)."""

    GLOBAL = "global"
    LOCAL = "local"


def _validated_sorted(spans) -> list[tuple[int, int, str]]:
    """Validate that ``spans`` exactly partitions [0, L); return them sorted.

    Raises ValueError on gaps, overlaps, non-zero start, zero-length or
    negative regions, and unknown scope names.
    """
    cleaned: list[tuple[int, int, str]] = []
    for span in spans:
        try:
            start, end, scope = span
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed span (want (start, end, scope)): {span!r}") from exc
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError(f"span bounds must be ints, got {span!r}")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"span bounds must be ints, got {span!r}")
        if start < 0:
            raise ValueError(f"span start must be >= 0, got {start}")
        if end <= start:
            raise ValueError(f"span must be non-empty, got ({start}, {end})")
        if scope not in (Scope.GLOBAL, Scope.LOCAL):
            raise ValueError(f"unknown scope {scope!r}; want {Scope.GLOBAL!r}/{Scope.LOCAL!r}")
        cleaned.append((start, end, scope))
    cleaned.sort()
    if cleaned and cleaned[0][0] != 0:
        raise ValueError(f"spans must partition starting at 0, first span starts at {cleaned[0][0]}")
    for prev, cur in zip(cleaned, cleaned[1:]):
        if cur[0] != prev[1]:
            kind = "overlap" if cur[0] < prev[1] else "gap"
            raise ValueError(f"spans do not partition [0, L): {kind} between {prev} and {cur}")
    return cleaned


def build_scope_tags(spans) -> list[str]:
    """Per-token scope tags of length L.

    Args:
        spans: List of ``(start, end_exclusive, scope)`` regions which must
            exactly partition ``[0, L)`` (order-insensitive).

    Returns:
        List of length L with Scope.GLOBAL / Scope.LOCAL per token.
        Empty spans list yields ``[]`` (L = 0 edge).

    Raises:
        ValueError: If the regions do not exactly partition [0, L).
    """
    ordered = _validated_sorted(spans)
    tags: list[str] = []
    for start, end, scope in ordered:
        tags.extend([scope] * (end - start))
    return tags


def same_local_scope(i: int, j: int, spans) -> bool:
    """True iff tokens i and j fall inside the same single LOCAL span."""
    ordered = _validated_sorted(spans)
    starts = [s for s, _, _ in ordered]

    def owner(tok: int):
        idx = bisect_right(starts, tok) - 1
        if idx < 0:
            return None
        s, e, scope = ordered[idx]
        return (s, e) if (s <= tok < e and scope == Scope.LOCAL) else None

    oi, oj = owner(i), owner(j)
    return oi is not None and oj is not None and oi == oj


def block_sparse_mask(spans) -> list[list[bool]]:
    """Full LxL boolean mask implementing Table 1 exactly.

    Row semantics: a GLOBAL row sees only GLOBAL columns (its KV-cache state
    is computed purely from other globals, insulating it from local edits);
    a LOCAL row sees all GLOBAL columns plus every column inside its own
    single LOCAL span, plus itself.
    """
    ordered = _validated_sorted(spans)
    if not ordered:
        return []
    total = ordered[-1][1]
    tags: list[str] = []
    owners: list[tuple[int, int]] = []
    for start, end, scope in ordered:
        tags.extend([scope] * (end - start))
        owners.extend([(start, end)] * (end - start))

    mask: list[list[bool]] = []
    for i in range(total):
        row: list[bool] = []
        tag_i = tags[i]
        own_i = owners[i]
        for j in range(total):
            if i == j:
                row.append(True)
            elif tag_i == Scope.LOCAL:
                # Bodies see signatures + coherent same-scope siblings.
                row.append(tags[j] == Scope.GLOBAL or owners[j] == own_i)
            else:  # GLOBAL row: insulated from every LOCAL column.
                row.append(tags[j] == Scope.GLOBAL)
        mask.append(row)
    return mask


def global_rows_frozen(mask_before, mask_after, spans_after) -> bool:
    """Assert-style check: are all GLOBAL rows unchanged vs ``mask_before``?

    A row tagged GLOBAL under ``spans_after`` must equal the corresponding
    ``mask_before`` row over the old extent and be False beyond it. Returns
    True/False instead of raising; shape mismatches simply read as False.
    Mirrors core's ``scoping.global_rows_frozen`` behavior exactly.
    """
    try:
        ordered = _validated_sorted(spans_after)
        total_new = ordered[-1][1] if ordered else 0
        total_old = len(mask_before)
        if total_new < total_old or len(mask_after) != total_new:
            return False
        if any(len(r) != total_new for r in mask_after):
            return False
        tags = build_scope_tags(ordered)
        for i, tag in enumerate(tags):
            if tag != Scope.GLOBAL:
                continue
            expected = list(mask_before[i]) + [False] * (total_new - total_old)
            if list(mask_after[i]) != expected:
                return False
        return True
    except (TypeError, ValueError, IndexError):
        return False


#: Alias matching the historical "frozen global rows/columns" wording used by
#: callers discussing the [EXPAND] cache-coherency guarantee.
frozen_global_rows = global_rows_frozen
