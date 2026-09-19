"""API 层测试 —— 守卫"服务会不会响应"这件事本身。

这层此前完全没有测试，而它的失效方式特别隐蔽：HTTP 服务不响应时
**没有任何报错**，只有客户端超时。实测就踩过一次：`do_GET` 里写成
`with _ENGINE_LOCK: engine = get_engine()`，而 `get_engine()` 内部也要拿
同一把**非重入**锁，于是 `/health` 自锁挂死 —— curl 只报超时，
日志里一个字都没有。

因此这里既测"锁的性质"（快、无需外部资源、能直接抓住死锁成因），
也测"端点真的能回话"（需要 Qdrant，用真实 HTTP 请求打一遍）。
"""

import json
import threading
import http.client
from http.server import ThreadingHTTPServer

import pytest

import api

pytestmark = pytest.mark.unit


class TestEngineLock:
    """锁的性质 —— 死锁的直接成因就在这里，必须有一条不依赖外部资源的守卫。"""

    def test_engine_lock_is_reentrant(self):
        """`_ENGINE_LOCK` 必须可重入。

        复现方式（修好之前）：把这里换成 `threading.Lock()`，然后跑
        `python api.py` 再 `curl /health` —— 会一直挂到超时。
        因为 do_GET 曾在持有该锁的情况下调用 get_engine()，而后者也要拿它。
        """
        lock = api._ENGINE_LOCK
        assert lock.acquire(timeout=0.5), "无法获取 _ENGINE_LOCK"
        try:
            acquired_again = lock.acquire(timeout=0.5)
            if acquired_again:
                lock.release()
            assert acquired_again, (
                "_ENGINE_LOCK 不可重入：任何'持锁时调用 get_engine()'的写法都会死锁"
            )
        finally:
            lock.release()

    def test_get_engine_is_idempotent(self):
        """引擎单例：重复调用拿到同一个对象，且不重复构造。"""
        a = api.get_engine()
        b = api.get_engine()
        assert a is b
        assert isinstance(a, __import__("rag_engine").RAGEngine)


def _get(server, path, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1],
                                      timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8")
    finally:
        conn.close()


def _post(server, path, body, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1],
                                      timeout=timeout)
    try:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        conn.request("POST", path, body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8")
    finally:
        conn.close()


@pytest.fixture
def server():
    """在临时端口起一个真实的 HTTP 服务（端口 0 = 由内核分配，不抢固定端口）。"""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


class TestRoutingAndValidation:
    """这两条不需要 Qdrant：路由与入参校验都发生在碰引擎之前。

    所以它们能进 L0 —— 快速、任何时候都能跑，适合当"服务还能回话"的哨兵。
    """

    def test_unknown_path_404(self, server):
        status, body = _get(server, "/nope")
        assert status == 404
        assert "未知路径" in json.loads(body)["error"]

    def test_health_is_reachable_not_deadlocked(self, server):
        """/health 至少要**回话**。

        这里不断言 200：Qdrant 没起时正确的响应是 503（且带可诊断的 error），
        而不是挂住。断言"有状态码返回"就能抓住死锁 —— 死锁时客户端只会超时。
        """
        status, body = _get(server, "/health", timeout=15)
        assert status in (200, 503), f"意外的状态码 {status}"
        data = json.loads(body)
        if status == 200:
            assert data["status"] in ("ok", "empty")
            assert "chunks" in data and "collection" in data
        else:
            assert "无法访问向量库" in data["error"]

    @pytest.mark.parametrize("path", ["/search", "/ask"])
    def test_bad_params_400(self, server, path, ):
        for bad in ({}, {"query": ""}, {"query": 123}, {"query": "x", "top_k": 0},
                    {"query": "x", "book": 5}):
            status, body = _post(server, path, bad)
            assert status == 400, f"{bad} 未被拦截，返回 {status}"
            assert "error" in json.loads(body)

    def test_invalid_json_400(self, server):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1],
                                          timeout=10)
        try:
            conn.request("POST", "/search", body=b"{not json",
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            assert resp.status == 400
            assert "JSON" in json.loads(resp.read().decode("utf-8"))["error"]
        finally:
            conn.close()

    def test_method_not_allowed(self, server):
        """未实现的 HTTP 方法不该 500。"""
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1],
                                          timeout=10)
        try:
            conn.request("DELETE", "/search")
            resp = conn.getresponse()
            assert resp.status in (405, 501), f"DELETE 返回了 {resp.status}"
            resp.read()
        finally:
            conn.close()
