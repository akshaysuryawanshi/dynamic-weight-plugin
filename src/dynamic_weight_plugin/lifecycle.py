"""Owned pause and deadline wakeups; model state stays on the engine thread."""

from functools import partial
from threading import Timer
from time import monotonic_ns
from weakref import ref


class DeadlineWakeup:
    def __init__(self, queue, wakeup):
        self.queue = queue
        self.wakeup = wakeup
        self.timer = None

    def clear(self):
        timer, self.timer = self.timer, None
        if timer is not None:
            timer.cancel()
            # Callback only puts a wakeup on the native unbounded input queue.
            # Joining prevents rapid operations from accumulating timer threads.
            timer.join(timeout=1)
            if timer.is_alive():
                self.timer = timer
                raise RuntimeError("Deadline wakeup thread did not stop")

    def arm(self, deadline_ns):
        self.clear()
        timer = Timer(max(0, (deadline_ns-monotonic_ns())/1e9),
                      self.queue.put_nowait, args=(self.wakeup,))
        timer.name = "dynamic-quantization-deadline"
        timer.daemon = True
        timer.start()
        self.timer = timer


class PauseLease:
    """Track the exact native idle callback and Future created by one pause."""
    def __init__(self, engine, callback):
        self.engine = engine
        self.callback = callback
        self.future = None
        self.owned = None
        self.active = True

    def start(self):
        before = {id(item) for item in self.engine._idle_state_callbacks}
        future = self.engine.pause_scheduler(mode="wait", clear_cache=False)
        if future is None:
            callback, self.callback = self.callback, None
            self.active = False
            callback()
            return
        self.future = future
        owned = [item for item in self.engine._idle_state_callbacks
                 if id(item) not in before and isinstance(item, partial)
                 and (item.keywords or {}).get("future") is future]
        if len(owned) != 1:
            # A changed native callback contract is unsafe to cancel/resume.
            raise RuntimeError("Unreviewed native pause callback ownership")
        self.owned = owned[0]
        # A retained native Future must not retain the engine/model after close.
        weak_lease = ref(self)
        def done(completed):
            lease = weak_lease()
            if lease is not None:
                lease._done(completed)
        future.add_done_callback(done)

    def _done(self, future):
        if not self.active:
            return
        self.active = False
        callback, self.callback = self.callback, None
        if self.owned is not None:
            callbacks = self.engine._idle_state_callbacks
            callbacks[:] = [item for item in callbacks if item is not self.owned]
        self.future = self.owned = None
        try:
            future.result()
        except Exception as exc:
            callback(exc)
        else:
            callback()

    def close(self):
        self.active = False
        self.callback = None
        future, owned = self.future, self.owned
        if future is not None and owned is None and not future.done():
            raise RuntimeError("Cannot release an unowned native pause callback")
        if owned is not None:
            callbacks = self.engine._idle_state_callbacks
            callbacks[:] = [item for item in callbacks if item is not owned]
        self.future = self.owned = None
        if future is not None and not future.done():
            # Detach before cancelling: Future callbacks execute synchronously.
            future.cancel()
