REDIS_URL = os.environ.get("REDIS_URL")
if not REDIS_URL:
    raise ValueError("❌ REDIS_URL environment variable is not set!")

QUEUE_NAME = os.getenv("RQ_QUEUE", "default")
JOB_STORAGE_ROOT = Path(os.getenv("JOB_STORAGE_ROOT", Path.cwd() / "jobs"))

redis_connection = redis.Redis.from_url(REDIS_URL)
job_queue = Queue(QUEUE_NAME, connection=redis_connection)
