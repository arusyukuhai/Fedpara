"""Bounded and ordered CPU/network input pipeline for HF PPO/SFT rows.

Prefetches *data only*. Never evaluates or samples the actor outside the adapter
lock: old policy log-probabilities cannot become stale before the step starts.
"""
from __future__ import annotations

import queue
import threading


class PrefetchedRows:
    """Single daemon producer, max two validated rows, ordered error delivery.

    The Hub iterator is owned exclusively by the producer. A blocked upstream
    network request may survive close until the transport times out; close must
    not block local trainer shutdown or issue an unsafe cross-thread iterator
    mutation. Exceptions propagate on the same next() they would without
    prefetch, preserving fail-closed training behavior.
    """
    def __init__(self, source, *, capacity=2, read_timeout=120.0):
        if type(capacity) is not int or not 1<=capacity<=8:
            raise ValueError('prefetch capacity must be 1..8')
        if not isinstance(read_timeout,(int,float)) or not 0.01<=read_timeout<=3600:
            raise ValueError('prefetch read timeout must be .01..3600 seconds')
        self.read_timeout=float(read_timeout)
        self.source=source
        self.queue=queue.Queue(maxsize=capacity)
        self.stop=threading.Event()
        self.closed=False
        self.worker=threading.Thread(target=self._produce,name='kagami-hub-prefetch',daemon=True)
        self.worker.start()

    def _produce(self):
        try:
            while not self.stop.is_set():
                try:
                    item=(True,self.source.next())
                except Exception as exc:
                    item=(False,exc)
                while not self.stop.is_set():
                    try:
                        self.queue.put(item,timeout=.1)
                        break
                    except queue.Full:
                        pass
                if not item[0]:
                    break
        finally:
            # Keep one owning thread for dataset.close even under failure.
            try:self.source.close()
            except Exception:pass

    def next(self):
        if self.closed:
            raise RuntimeError('Dataset prefetcher was closed')
        try:
            valid,result=self.queue.get(timeout=self.read_timeout)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError('Hub dataset prefetch stalled; check network, resume the trainer from its last durable checkpoint') from exc
        if not valid:
            self.close()
            raise result
        return result

    def close(self):
        self.closed=True
        self.stop.set()
        # Do not block on a hung remote dataset read during shutdown.
        self.worker.join(timeout=.05)
