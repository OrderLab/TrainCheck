"""Write back list slots that an unproxied callee replaces (issue #176).

Unproxying copies list arguments, so the caller's lists keep their own elements:
an optimizer's ``params`` list must keep the proxied parameters, because
``optimizer.state`` is keyed by them. A few APIs instead return results by
replacing slots of a list argument. ``torch.optim.sgd.sgd`` stores each newly
created momentum buffer into ``momentum_buffer_list[i]``, and ``SGD.step``
reads the buffers back from its own list after the call. With the copy, those
writes were lost and SGD momentum was silently reset every step.

The registry below names such arguments. For a registered argument passed as a
list, the callee receives the unproxied copy as before; after the call, every
slot the callee replaced (compared by identity, so ``None`` becoming a tensor
counts) is written into the caller's list. Slots the callee did not replace,
including slots holding proxies, are left untouched. The write-back also runs
when the callee raises, so the caller sees the same partial updates it would
see without instrumentation. Registered lists must keep their length, must not
be passed to two registered parameters at once, and must not have a replaced
slot changed through another reference during the call; each of these is
reported as an error instead of losing or overwriting writes silently.
"""

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

# Fully qualified API name, as produced by ``traincheck.utils.typename``, mapped
# to the names of its fixed-length list parameters whose replaced slots must be
# written back to the caller.
LIST_WRITEBACK_ARGUMENTS: Mapping[str, frozenset[str]] = {
    "torch.optim.sgd.sgd": frozenset({"momentum_buffer_list"}),
}


class ListWritebackError(RuntimeError):
    """A registered list argument cannot be written back as registered."""


# Registered parameter name mapped to its positional index, or ``None`` when it
# can only be passed by keyword.
ListWritebackPositions = Mapping[str, int | None]


@dataclass(frozen=True)
class PendingListWriteback:
    """One registered list argument of one call, paired with the copy the callee gets."""

    api_name: str
    parameter: str
    caller_list: list[object]
    caller_before: tuple[object, ...]
    callee_list: list[object]
    passed: tuple[object, ...]


def has_list_writeback(api_name: str) -> bool:
    """Whether calls to ``api_name`` have registered list arguments."""
    return api_name in LIST_WRITEBACK_ARGUMENTS


def list_writeback_positions(
    function: Callable[..., object], api_name: str
) -> ListWritebackPositions | None:
    """Resolve the registered parameters of ``function``; ``None`` if it has none.

    Call this once when wrapping ``function``. Raises ``ListWritebackError``
    when the registry and the installed library disagree, so a renamed
    parameter cannot silently disable the write-back.
    """
    parameters = LIST_WRITEBACK_ARGUMENTS.get(api_name)
    if parameters is None:
        return None
    return _parameter_positions(function, api_name, parameters)


def _parameter_positions(
    function: Callable[..., object], api_name: str, parameters: frozenset[str]
) -> dict[str, int | None]:
    """Positional index of each registered parameter, ``None`` if keyword-only."""
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as error:
        raise ListWritebackError(
            f"{api_name} has registered list arguments but no inspectable signature"
        ) from error
    names = list(signature.parameters)
    positions: dict[str, int | None] = {}
    for parameter in sorted(parameters):
        if parameter not in signature.parameters:
            raise ListWritebackError(
                f"{api_name} has no parameter {parameter!r} registered for list write-back"
            )
        kind = signature.parameters[parameter].kind
        if kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positions[parameter] = names.index(parameter)
        elif kind is inspect.Parameter.KEYWORD_ONLY:
            positions[parameter] = None
        else:
            raise ListWritebackError(
                f"{api_name} parameter {parameter!r} is variadic and cannot be registered "
                "for list write-back"
            )
    return positions


def prepare_list_writebacks(
    api_name: str,
    positions: ListWritebackPositions,
    caller_args: Sequence[object],
    caller_kwargs: Mapping[str, object],
    callee_args: Sequence[object],
    callee_kwargs: Mapping[str, object],
) -> list[PendingListWriteback]:
    """Pair each registered list argument of this call with its unproxied copy.

    ``positions`` comes from ``list_writeback_positions``.
    ``caller_args``/``caller_kwargs`` are the arguments the caller passed and
    ``callee_args``/``callee_kwargs`` the unproxied ones the callee will
    receive, in the same positions. A registered argument that was not passed,
    or was not passed as a list, needs no write-back: the callee cannot replace
    slots of the caller's object through it, so ordinary unproxying stays in
    effect.
    """
    pending: list[PendingListWriteback] = []
    for parameter in sorted(positions):
        position = positions[parameter]
        if parameter in caller_kwargs:
            caller_value = caller_kwargs[parameter]
            callee_value = callee_kwargs[parameter]
        elif position is not None and position < len(caller_args):
            caller_value = caller_args[position]
            callee_value = callee_args[position]
        else:
            continue
        # The same exact-type test as unproxy_arg, which copies exact lists
        # only; any other object, list subclasses included, reaches the callee
        # as the caller's own. isinstance narrows the type for the checker.
        if not isinstance(caller_value, list) or type(caller_value) not in [list]:
            continue
        assert (
            isinstance(callee_value, list) and callee_value is not caller_value
        ), f"{api_name}: argument {parameter!r} was not copied by unproxying"
        for other in pending:
            if other.caller_list is caller_value:
                # Unproxying gives each occurrence its own copy, so the callee
                # cannot see the two parameters as one list.
                raise ListWritebackError(
                    f"{api_name} received the same list as {other.parameter!r} and "
                    f"{parameter!r}; aliased registered lists cannot be written back"
                )
        pending.append(
            PendingListWriteback(
                api_name=api_name,
                parameter=parameter,
                caller_list=caller_value,
                caller_before=tuple(caller_value),
                callee_list=callee_value,
                passed=tuple(callee_value),
            )
        )
    return pending


def apply_list_writebacks(pending: Sequence[PendingListWriteback]) -> None:
    """Write every slot the callee replaced back into the caller's list.

    Every list is checked before any slot is written, so an error leaves the
    caller's lists as the callee left them.
    """
    replaced: list[tuple[PendingListWriteback, int, object]] = []
    for item in pending:
        expected = len(item.passed)
        if len(item.callee_list) != expected or len(item.caller_list) != expected:
            raise ListWritebackError(
                f"{item.api_name} changed the length of its list argument "
                f"{item.parameter!r} (passed {expected} elements, callee now has "
                f"{len(item.callee_list)}, caller now has {len(item.caller_list)}); "
                "the replaced slots cannot be written back"
            )
        for index, (before, after) in enumerate(zip(item.passed, item.callee_list)):
            if after is before:
                continue
            current = item.caller_list[index]
            if current is not item.caller_before[index] and current is not after:
                raise ListWritebackError(
                    f"{item.api_name} replaced slot {index} of {item.parameter!r}, "
                    "and the caller's list was changed at that slot through another "
                    "reference during the call; the write-back would overwrite it"
                )
            replaced.append((item, index, after))
    for item, index, after in replaced:
        item.caller_list[index] = after


def call_with_list_writebacks(
    function: Callable[..., object],
    pending: Sequence[PendingListWriteback],
    args: Sequence[object],
    kwargs: Mapping[str, object],
) -> object:
    """Call ``function`` and apply ``pending`` write-backs, also when it raises."""
    if not pending:
        return function(*args, **kwargs)
    try:
        result = function(*args, **kwargs)
    except BaseException as error:
        # Without instrumentation the caller would see the slots replaced
        # before the exception, so write them back and re-raise unchanged.
        try:
            apply_list_writebacks(pending)
        except ListWritebackError as writeback_error:
            raise writeback_error from error
        raise
    apply_list_writebacks(pending)
    return result
