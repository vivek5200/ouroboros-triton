"""Module 6: Supervisor-Worker HA Serving Layer.

CONSTRAINT: The Supervisor process owns the KV-cache and block table.
Workers are stateless and restartable via SupervisorD.
"""


class Supervisor:
    """Supervisor process that owns KV-cache and block table.

    TODO: Implement full supervisor with health monitoring.
    """

    def __init__(self):
        self.workers: list[int] = []
        self.is_running = False

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False


class Worker:
    """Stateless inference worker, restartable by SupervisorD.

    TODO: Implement stateless inference serving.
    """

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.is_alive = False

    def start(self) -> None:
        self.is_alive = True

    def stop(self) -> None:
        self.is_alive = False
