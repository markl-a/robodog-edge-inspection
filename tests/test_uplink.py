from edge.uplink import Outbox, result_key


class FlakyPlatform:
    """Drops the connection on some calls; on others stores the batch but the ACK is lost."""
    def __init__(self, script):
        self.script, self.stored, self.calls = list(script), {}, 0

    def __call__(self, rows):
        self.calls += 1
        mode = self.script.pop(0) if self.script else "ok"
        if mode == "down":
            raise ConnectionError("no route to host")
        for r in rows:
            self.stored.setdefault(r["idempotencyKey"], r)      # server-side dedupe on the key
        if mode == "lost_ack":
            raise TimeoutError("ack lost")
        return {r["idempotencyKey"] for r in rows}


def rows(n):
    return [{"idempotencyKey": result_key("m", f"cp{i}"), "checkpointId": f"cp{i}"} for i in range(n)]


def test_results_survive_outage_and_arrive_exactly_once():
    ob = Outbox()
    for r in rows(120):
        ob.put(r)
    srv = FlakyPlatform(["down", "lost_ack", "ok", "down"])
    for _ in range(6):
        ob.flush(srv, batch=50)
    assert ob.pending() == []
    assert len(srv.stored) == 120


def test_reinspection_replaces_pending_result():
    ob = Outbox()
    ob.put({"idempotencyKey": "k", "value": 1})
    ob.put({"idempotencyKey": "k", "value": 2})
    assert ob.pending() == [{"idempotencyKey": "k", "value": 2}]
