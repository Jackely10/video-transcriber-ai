import server


def test_healthz_returns_status_ok(monkeypatch):
    class DummyRedis:
        def ping(self) -> bool:
            return True

    def fake_connection():
        return DummyRedis()

    class DummyQueue:
        def __init__(self, name, connection=None):
            self.count = 3

    monkeypatch.setattr(server, "_get_redis_connection", fake_connection)
    monkeypatch.setattr(server, "Queue", DummyQueue)

    client = server.app.test_client()
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["redis"] is True
    assert payload["worker_queue_len"] == 3


def test_protected_route_requires_basic_auth_when_enabled(monkeypatch):
    monkeypatch.setenv("BASIC_AUTH_USERNAME", "demo")
    monkeypatch.setenv("BASIC_AUTH_PASSWORD", "secret")

    client = server.app.test_client()
    response = client.get("/job/test-id")
    assert response.status_code == 401
