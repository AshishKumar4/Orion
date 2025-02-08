import queue
import threading
from concurrent.futures import Executor, Future
from typing import Iterator, Any, Optional, Callable

class BackgroundStream:
    """
    Consumes a generator within the provided Executor's worker thread,
    storing items in a thread-safe queue. The main thread can iterate
    over this instance to retrieve the generated items.

    No new thread is created beyond what the Executor manages.
    """

    def __init__(self, generator_fn: Callable[[], Iterator[Any]], executor: Executor):
        """
        :param generator_fn: A callable returning the generator to consume.
        :param executor: The executor in which to run the generator consumption.
        """
        self._queue: queue.Queue[Optional[Any]] = queue.Queue()
        self._stop_event = threading.Event()
        self._future: Future = executor.submit(self._run, generator_fn)

    def _run(self, generator_fn: Callable[[], Iterator[Any]]) -> None:
        """
        Worker function that runs inside the Executor's worker thread.
        It pulls from the generator, enqueues items, then enqueues None
        when finished or interrupted.
        """
        try:
            gen = generator_fn()
            for item in gen:
                if self._stop_event.is_set():
                    break
                self._queue.put(item)
        finally:
            # Signal iteration is over
            self._queue.put(None)

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over the buffered items. This will block if the queue is empty
        but the generator is still producing items.
        """
        while True:
            item = self._queue.get()
            if item is None:
                break
            yield item

    def stop(self):
        """
        Signal that we want to stop consuming the generator early.
        This sets _stop_event, so the worker thread stops pulling items,
        and enqueues None to unblock any current iteration.
        """
        self._stop_event.set()
        self._queue.put(None)
        # Optionally, you might also cancel the future:
        if not self._future.done():
            self._future.cancel()
