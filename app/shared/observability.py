"""Logging setup and request-id propagation."""

import logging
import re
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

# Each request runs in its own task, so the value never leaks between requests.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

logger = logging.getLogger(__name__)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s"
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id and logs one access line per request.

    The query string is not logged: search queries may contain client PII.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        request_id_var.set(request_id)

        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        finally:
            logger.info(
                "request method=%s path=%s status=%s duration_ms=%.1f",
                request.method,
                request.url.path,
                status_code,
                (time.perf_counter() - started) * 1000,
            )
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
