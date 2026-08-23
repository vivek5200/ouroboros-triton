"""Module 6 §8: Supervisor-Worker HA Serving Layer.

CONSTRAINT: The Supervisor process owns the KV-cache and block table.
Workers are stateless and restartable via SupervisorD.

MODULE-6 MAPPING (paper §8, Supervisor-Worker HA)
-------------------------------------------------
Ownership:
    * The Supervisor is the single owner of THE ``BlockTable`` (Module 2
      page-table / VRAM bookkeeping) and of ``self.kv_cache`` (dict).
      Worker objects never hold references to either; the only door from
      the compute side into supervisor-owned state is a supervisor
      method — minimally ``Supervisor.table_op(fn)`` (and, for tests,
      direct attribute reads on the *supervisor*, which are fine: the
      rule constrains workers, not the operator).
Statelessness & restart:
    * ``Worker.process()`` is a pure function of ``(alive, request)``:
      it writes nothing into the worker (verified by test), so a Worker
      object carries no per-request state. ``WorkerCrash`` simulates a
      Triton-kernel segfault killing the worker instantly. Restarting
      therefore means "drop the dead object, construct a fresh one"
      (``Supervisor.restart``) with ZERO state migration — there is no
      state to migrate.
Blast radius & Phase-1 Safe Queue:
    * A segfault destroys exactly the one request in flight
      (blast radius = one request, not the node). ``submit()`` catches
      it, marks the worker dead, appends the raw request to
      ``self.safe_queue`` and RETURNS the sentinel string
      ``SAFE_QUEUE_PREFIX + request`` instead of raising.
      DOCUMENTED CHOICE: submit degrades rather than raises — the paper's
      contract is degradation to the Phase-1 PyTorch "Safe Queue", so the
      caller observes a distinguishable result (prefix check /
      ``drain_safe_queue``) while the serving loop keeps running.
      ``drain_safe_queue()`` replays requests FIFO exactly once through
      ``submit()`` itself, so each drained request is re-dispatched to a
      live worker and allocated a FRESH chain (the Phase-1 fallback path
      re-serves them; nothing is ever served from a stale chain).
Resource semantics (paper §4.2: sequences = chains of 64-token blocks):
    * Every ACCEPTED request allocates a supervisor-owned chain sized by
      its payload: ``n_blocks = max(1, ceil(byte_len / 64))`` where
      ``byte_len = len(request.encode("utf-8"))``. DOCUMENTED CHOICE:
      "length" means UTF-8 BYTE length (the wire payload the KV-cache
      must hold), not Python character count.
    * ``kv_cache[request_id] = head`` records ownership; ``request_id``
      comes from a monotonic counter exposed as ``last_request_id``
      (starts at 0, advances on every accepted submit — including ones
      that later crash, so ids are never reused).
    * On WorkerCrash mid-flight the chain is freed FIRST — every block
      of ``walk(head)`` materialized then freed, because ``free_block``
      clears ``next_ptr`` and would truncate a lazy walk — BEFORE the
      request is safe-queued. Crash leaks ZERO blocks.
    * ``expand_request(request_id, extra_bytes)`` is the supervisor-side
      [EXPAND]: walk to the recorded tail and ``expand_chain`` exactly
      ``ceil(extra_bytes / 64)`` times; each new block adds 64 tokens of
      capacity, tracked in ``self.chain_capacity[request_id]``.
    * ``release(request_id)`` frees the whole chain and drops both cache
      entries; double release raises RuntimeError.
"""

from typing import Callable

from .block_table import BLOCK_SIZE, BlockTable

#: Prefix marking a request degraded to the Phase-1 Safe Queue by submit().
SAFE_QUEUE_PREFIX = "safe-queued:"


class WorkerCrash(RuntimeError):
    """Simulated Triton-kernel segfault: kills the worker mid-request."""


class Worker:
    """Stateless inference worker; restartable because it owns nothing.

    Attributes:
        worker_id: Stable identity handed in at construction (survives
            restarts as the routing key; carries no runtime state).
        alive: Lifecycle flag only — ``False`` after ``kill()`` or after a
            ``fail_next`` crash fires.

    Statelessness enforcement: ``process()`` is pure — on success it
    mutates no instance attribute (see ``tests/test_serving.py``), and its
    output depends solely on the request string. The only mutation in the
    class is consuming the one-shot ``fail_next`` death sentence, which is
    lifecycle bookkeeping, not request state.
    """

    def __init__(self, worker_id, fail_next: bool = False):
        self.worker_id = worker_id
        self.alive = False
        self._fail_next = bool(fail_next)

    def start(self) -> None:
        """Bring the worker up (post-spawn)."""
        self.alive = True

    def kill(self) -> None:
        """Simulated segfault: the worker dies instantly."""
        self.alive = False

    def process(self, request: str) -> str:
        """Deterministic stand-in for one inference step.

        Pure transform: ``"ok:<request reversed>"``. Raises ``WorkerCrash``
        if this worker is dead (a caller poking a corpse) or if the armed
        one-shot ``fail_next`` flag fires, which also kills the worker —
        a segfault takes the process down mid-request.
        """
        if not self.alive:
            raise WorkerCrash(f"worker {self.worker_id!r} is dead (segfaulted)")
        if self._fail_next:
            self._fail_next = False  # one-shot consumed
            self.kill()
            raise WorkerCrash(
                f"worker {self.worker_id!r} simulated segfault (fail_next)"
            )
        return f"ok:{request[::-1]}"


class Supervisor:
    """Supervisor process: owns KV-cache + block table; workers are cattle.

    All worker access to supervisor-owned state goes through supervisor
    methods (``table_op`` here); workers receive no references at spawn.
    """

    def __init__(self, max_blocks: int = 1024):
        # THE shared Module-2 table + THE KV-cache: supervisor-owned, always.
        self.block_table = BlockTable(max_blocks)
        self.kv_cache: dict = {}  # request_id -> chain head (physical block id)
        # request_id -> chain capacity in tokens (n_blocks * 64); [EXPAND] bookkeeping.
        self.chain_capacity: dict = {}
        # worker_id -> Worker, insertion order = routing priority.
        self.workers: dict = {}
        self.safe_queue: list[str] = []  # Phase-1 PyTorch fallback queue
        self._request_counter = 0  # monotonic request-id source

    @property
    def last_request_id(self) -> int:
        """Most recently assigned request id (0 before the first submit)."""
        return self._request_counter

    # -- chain lifecycle (supervisor-owned VRAM bookkeeping) -------------

    @staticmethod
    def _payload_blocks(request: str) -> int:
        """Blocks needed for a request payload: max(1, ceil(byte_len / 64)).

        "Length" is the UTF-8 BYTE length of the request string (documented
        module choice) — the payload the KV-cache must physically hold.
        """
        if not isinstance(request, str):
            raise RuntimeError(
                f"request must be str, got {type(request).__name__}"
            )
        byte_len = len(request.encode("utf-8"))
        return max(1, -(-byte_len // BLOCK_SIZE))  # ceil, min 1 block

    def _free_chain(self, request_id) -> None:
        """Free every block of the chain owned by ``request_id`` + bookkeeping.

        The walk is MATERIALIZED first: ``free_block`` clears both the slot
        payload and ``next_ptr``, so freeing lazily would truncate the walk
        and orphan the rest of the chain (a leak).
        """
        head = self.kv_cache.pop(request_id)
        self.chain_capacity.pop(request_id, None)
        for block_id in list(self.block_table.walk(head)):
            self.block_table.free_block(block_id)

    def expand_request(self, request_id, extra_bytes: int) -> int:
        """Supervisor-side [EXPAND]: grow a live chain to cover ``extra_bytes``.

        Walks the recorded chain to its tail and calls ``expand_chain`` exactly
        ``ceil(extra_bytes / 64)`` times (each new block adds 64 tokens of
        capacity), updating ``chain_capacity[request_id]``.

        Returns:
            The NEW tail block id.

        Raises:
            RuntimeError: unknown/already-released ``request_id``, non-int or
                non-positive ``extra_bytes``, or the table is out of blocks
                (propagated from ``expand_chain`` with table state unchanged).
        """
        if request_id not in self.kv_cache:
            raise RuntimeError(
                f"unknown or already-released request id: {request_id!r}"
            )
        if isinstance(extra_bytes, bool) or not isinstance(extra_bytes, int):
            raise RuntimeError(f"invalid extra_bytes: {extra_bytes!r}")
        if extra_bytes <= 0:
            raise RuntimeError(f"extra_bytes must be > 0, got {extra_bytes}")
        head = self.kv_cache[request_id]
        tail = head
        for block_id in self.block_table.walk(head):
            tail = block_id
        n_new = -(-extra_bytes // BLOCK_SIZE)  # ceil(extra_bytes / 64)
        for _ in range(n_new):
            tail = self.block_table.expand_chain(tail)
            self.chain_capacity[request_id] += BLOCK_SIZE
        return tail

    def release(self, request_id) -> None:
        """Free the whole chain owned by ``request_id`` and drop cache entries.

        Raises:
            RuntimeError: unknown id or double release.
        """
        if request_id not in self.kv_cache:
            raise RuntimeError(
                f"unknown or already-released request id: {request_id!r}"
            )
        self._free_chain(request_id)

    # -- workforce management ------------------------------------------

    def add_worker(self, worker_id, fail_next: bool = False) -> Worker:
        """Spawn a fresh alive worker under the given id."""
        if worker_id in self.workers:
            raise RuntimeError(f"worker {worker_id!r} already exists")
        w = Worker(worker_id, fail_next=fail_next)
        w.start()
        self.workers[worker_id] = w
        return w

    def health_pass(self) -> list:
        """Ids of dead workers from the latest sweep."""
        return [wid for wid, w in self.workers.items() if not w.alive]

    def restart(self, worker_id) -> Worker:
        """Replace a worker with a fresh alive one.

        Stateless workers make this trivially safe: no state migration,
        just drop the dead object and construct a new one under the same
        routing id.
        """
        if worker_id not in self.workers:
            raise RuntimeError(f"unknown worker id: {worker_id!r}")
        fresh = Worker(worker_id)
        fresh.start()
        self.workers[worker_id] = fresh  # old object simply garbage-collected
        return fresh

    # -- supervisor-mediated state access (the ONLY door) ---------------

    def table_op(self, fn: Callable[..., object], *args, **kwargs) -> object:
        """Run ``fn(block_table, *args, **kwargs)`` against THE shared table.

        This passthrough is how any compute-side code touches the Module-2
        page table without ever holding a private reference to it.
        """
        return fn(self.block_table, *args, **kwargs)

    # -- request routing -------------------------------------------------

    def submit(self, request: str, worker_id=None) -> str:
        """Route one request; degrade to Phase-1 Safe Queue on segfault.

        An accepted request first allocates a supervisor-owned chain of
        ``max(1, ceil(byte_len/64))`` blocks and records
        ``kv_cache[request_id] = head`` (``request_id`` from the monotonic
        counter; see ``last_request_id``, ``chain_capacity``).

        Returns ``"ok:<reversed>"`` on success, or
        ``SAFE_QUEUE_PREFIX + request`` when the worker crashed handling
        it (documented sentinel choice — see module docstring): the crash
        is contained, the in-flight chain is freed FIRST (zero block leak),
        THEN the request is queued FIFO, the worker is marked dead, and the
        caller keeps running.
        """
        if worker_id is not None:
            if worker_id not in self.workers:
                raise RuntimeError(f"unknown worker id: {worker_id!r}")
            candidates = [worker_id]
        else:
            candidates = list(self.workers)
        worker = next(
            (self.workers[w] for w in candidates if self.workers[w].alive),
            None,
        )
        if worker is None:
            raise RuntimeError("no alive worker available")
        # Accept: allocate the supervisor-owned chain BEFORE dispatch so the
        # crash path below owns exactly what it must free back.
        self._request_counter += 1  # monotonic; never reused, even after a crash
        request_id = self._request_counter
        head = self.block_table.allocate_chain(self._payload_blocks(request))
        self.kv_cache[request_id] = head
        self.chain_capacity[request_id] = (
            self.block_table.chain_len(head) * BLOCK_SIZE
        )
        try:
            return worker.process(request)
        except WorkerCrash:
            # Blast radius = this one request: mark dead, free the in-flight
            # chain (BEFORE safe-queueing — a crash must not leak blocks),
            # degrade it to the Phase-1 Safe Queue, keep the loop alive.
            worker.kill()
            self._free_chain(request_id)
            self.safe_queue.append(request)
            return SAFE_QUEUE_PREFIX + request

    def drain_safe_queue(self) -> list[str]:
        """Re-dispatch Phase-1 Safe Queue contents FIFO through ``submit()``.

        Each drained request gets a FRESH alive worker and a FRESHLY
        allocated chain (never a stale one). Returns the list of final
        results in FIFO order and clears the queue exactly once.

        DOCUMENTED CHOICE: if no worker is alive at drain time, submit()
        would raise — instead the request is re-queued (never lost) and its
        reported result is the degraded sentinel ``SAFE_QUEUE_PREFIX +
        request``. A re-dispatch that itself crashes goes right back onto
        the queue with the same sentinel, so repeated drains make progress
        as soon as any worker is restarted.
        """
        drained = self.safe_queue[:]
        self.safe_queue.clear()
        results: list[str] = []
        for request in drained:
            try:
                results.append(self.submit(request))
            except RuntimeError:
                # Only escape from submit(): "no alive worker available".
                # (WorkerCrash is contained inside submit itself.)
                self.safe_queue.append(request)
                results.append(SAFE_QUEUE_PREFIX + request)
        return results
