from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from .common import BridgeError, ModelError, canonical, strict_json


class BoundedHTTPServer(HTTPServer):
    """Fixed worker count, bounded admission; use behind an authenticated proxy."""

    def __init__(self, address, handler, workers=4):
        super().__init__(address, handler)
        self.pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="nlbridge"
        )
        self.slots = threading.BoundedSemaphore(workers)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
            finally:
                self.shutdown_request(request)
            return

        def work():
            try:
                self.finish_request(request, client_address)
            except Exception:
                self.handle_error(request, client_address)
            finally:
                self.shutdown_request(request)
                self.slots.release()

        self.pool.submit(work)

    def server_close(self):
        super().server_close()
        self.pool.shutdown(wait=True)


def make_server(runtime, host="127.0.0.1", port=8080, workers=4):
    if not 1 <= workers <= 64:
        raise BridgeError("workers must be 1..64")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def log_message(self, *args):
            pass

        def respond(self, status, body):
            data = canonical(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True

        def do_GET(self):
            if self.path != "/health":
                return self.respond(404, {"error": "not_found"})
            self.respond(
                200,
                {"status": "ok", "catalog_revision": runtime.store.snapshot.revision},
            )

        def do_POST(self):
            if self.path != "/v1/compile":
                return self.respond(404, {"error": "not_found"})
            try:
                if (
                    self.headers.get("Transfer-Encoding")
                    or len(self.headers.get_all("Content-Length", [])) != 1
                ):
                    raise BridgeError(
                        "one Content-Length required; chunked input unsupported"
                    )
                try:
                    size = int(self.headers["Content-Length"])
                except ValueError:
                    raise BridgeError("invalid Content-Length")
                if size <= 0 or size > 65536:
                    return self.respond(413, {"error": "request exceeds 64 KiB"})
                data = strict_json(self.rfile.read(size))
                if (
                    not isinstance(data, dict)
                    or set(data) - {"text", "context", "args"}
                    or "text" not in data
                ):
                    raise BridgeError(
                        "expected text and optional context/args; policy is server-owned"
                    )
                self.respond(200, runtime.compile(**data))
            except ModelError as e:
                self.respond(502, {"error": "model_error", "message": str(e)})
            except (BridgeError, UnicodeError) as e:
                self.respond(400, {"error": "invalid_request", "message": str(e)})
            except Exception:
                self.respond(500, {"error": "internal_error"})

    return BoundedHTTPServer((host, port), Handler, workers)
