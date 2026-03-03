"""Decorators for stopping for-loops early.

Allow for decorating ml/tune and ml/train and stopping long running
jobs while saving the current information with Ctrl+C.
"""

import functools
import signal
import threading


def break_loop(func):
    """Interrupt-friendly decorator.

    The wrapped function receives the wrapper itself as its first argument,
    allowing it to query ``wrapper.stop_event`` (a :class:`threading.Event`)
    to terminate a long‑running loop after the current iteration when the user
    presses **Ctrl+C** once. A second **Ctrl+C** aborts immediately.

    Parameters
    ----------
    func : callable
        The function to be wrapped. It must accept the wrapper instance as its
        first positional argument, e.g.:

        .. code-block:: python

            @break_loop
            def process(wrapper, data):
                for i in grange(len(data), stop_event=wrapper.stop_event):
                    # do work ...

    Returns
    -------
    callable
        The wrapped function. The wrapper has an attribute ``stop_event``
        (a :class:`threading.Event`) that can be inspected inside ``func``.

    Notes
    -----
    * The decorator temporarily replaces the process‑wide ``SIGINT`` handler.
      The original handler is restored after the function finishes (or raises).
    * ``wrapper.stop_event`` is cleared before each call so that the same
      decorated function can be invoked repeatedly.

    Examples
    --------
    >>> @break_loop
    ... def count(wrapper, n):
    ...     for i in grange(n, stop_event=wrapper.stop_event):
    ...         print(i)
    ...         time.sleep(0.5)      # simulate work
    ...
    >>> count(10)         # Press Ctrl+C once to stop after the current
    ...                   # iteration; press twice to raise KeyboardInterrupt
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.stop_event = threading.Event()
        wrapper.stop_event.clear()
        interrupt_count = 0

        def handler(sig, frame):
            nonlocal interrupt_count
            interrupt_count += 1

            if interrupt_count == 1:
                print("\nCtrl+C detected. Will exit after current iteration...")
                wrapper.stop_event.set()
            else:
                print("\nSecond Ctrl+C detected. Forcing quit.")
                raise KeyboardInterrupt

        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, handler)

        try:
            _ = func(wrapper, *args, **kwargs)
        finally:
            signal.signal(signal.SIGINT, original_handler)
            wrapper.stop_event.clear()

    return wrapper


def grange(*args, stop_event):
    """Interrupt-friendly generator (``range`` type).

    Parameters
    ----------
    *args : int
        The same positional arguments accepted by the built‑in :func:`range`.
        Typical signatures are ``stop``, ``start, stop`` or
        ``start, stop, step``.
    stop_event : ``threading.Event``
        An event object whose ``is_set`` method is consulted before each
        iteration. If the event is set, the generator stops early.

    Yields
    ------
    int
        Values produced by the underlying ``range`` call.

    Raises
    ------
    TypeError
        If ``stop_event`` is not provided as a keyword argument.

    Notes
    -----
    This helper is intended to be used together with a function wrapped by
    :func:`break_loop`, which supplies the ``stop_event`` tied to a SIGINT
    handler.

    Examples
    --------
    >>> ev = threading.Event()
    >>> for i in grange(0, 10, 2, stop_event=ev):
    ...     print(i)
    0
    2
    4
    6
    8
    """
    for value in range(*args):
        if stop_event.is_set():
            print("Breaking loop...")
            break
        yield value


def finish_iteration(func):
    """Interrupt-friendly decorator.

    The first ``Ctrl+C`` (SIGINT) request sets a flag; after the wrapped
    function returns, a :class:`KeyboardInterrupt` is raised if the flag was
    set. A second ``Ctrl+C`` raises ``KeyboardInterrupt`` immediately.

    Parameters
    ----------
    func : callable
        The function to be wrapped. Its signature is unchanged.

    Returns
    -------
    callable
        The wrapped function that respects the graceful‑interrupt semantics.

    Notes
    -----
    * The decorator temporarily replaces the process‑wide ``SIGINT`` handler.
      The original handler is restored after ``func`` completes.
    * If the wrapped function itself raises ``KeyboardInterrupt``,
      the exception propagates unchanged.

    Examples
    --------
    >>> @finish_iteration
    ... def heavy_task():
    ...     for i in range(100):
    ...         print(i)
    ...         time.sleep(0.1)
    ...
    >>> heavy_task()              # Press Ctrl+C once to finish the current loop
    ...                           # iteration then raise KeyboardInterrupt.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        exit_requested = False
        interrupt_count = 0

        def handler(sig, frame):
            nonlocal exit_requested, interrupt_count
            interrupt_count += 1

            if interrupt_count == 1:
                print("\nCtrl+C detected. Will break loop after current iteration...")
                exit_requested = True
            else:
                print("\nSecond Ctrl+C detected. Forcing quit.")

        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, handler)

        try:
            result = func(*args, **kwargs)
        finally:
            signal.signal(signal.SIGINT, original_handler)

        if exit_requested:
            raise KeyboardInterrupt

        return result

    return wrapper
