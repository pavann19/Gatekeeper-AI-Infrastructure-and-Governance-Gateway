"""Bound HTTP body memory and receive time before JSON parsing."""
import asyncio
import time

from starlette.responses import JSONResponse

from core.config import settings
from core.request_budget import deadline


class RequestIngress:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] not in ('POST', 'PUT', 'PATCH'):
            await self.app(scope, receive, send)
            return
        messages = []
        size = 0
        lengths = [value.strip() for key, value in scope.get('headers', []) if key.lower() == b'content-length']
        if lengths:
            if len(lengths) != 1 or not lengths[0].isdigit() or len(lengths[0]) > 20:
                await JSONResponse({'detail': 'Invalid Content-Length'}, status_code=400)(scope, receive, send)
                return
            if int(lengths[0]) > settings.REQUEST_MAX_BODY_BYTES:
                await JSONResponse({'detail': 'Request body exceeds byte limit'}, status_code=413)(scope, receive, send)
                return
        end = deadline.get()
        if end is None:
            end = time.monotonic() + settings.ASSESS_TIMEOUT_SECONDS
        while True:
            try:
                message = await asyncio.wait_for(receive(), max(0, end - time.monotonic()))
            except TimeoutError:
                await JSONResponse({'detail': 'Request body deadline exhausted'}, status_code=408)(scope, receive, send)
                return
            if message['type'] == 'http.disconnect':
                return
            size += len(message.get('body', b''))
            if size > settings.REQUEST_MAX_BODY_BYTES:
                await JSONResponse({'detail': 'Request body exceeds byte limit'}, status_code=413)(scope, receive, send)
                return
            messages.append(message)
            if not message.get('more_body'):
                break
        position = 0
        async def replay():
            nonlocal position
            if position < len(messages):
                value = messages[position]
                position += 1
                return value
            return await receive()
        await self.app(scope, replay, send)
