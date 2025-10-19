from __future__ import annotations

import logging
import os

from rq import SimpleWorker, Worker

from jobs import QUEUE_NAME, redis_connection


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if os.name == "nt":
        worker = SimpleWorker([QUEUE_NAME], connection=redis_connection)
    else:
        worker = Worker([QUEUE_NAME], connection=redis_connection)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
