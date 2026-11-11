"""Token-scoped HTTP endpoints for the local review application."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import mimetypes
from urllib.parse import unquote, urlsplit
import uuid


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, app):
        self.app = app
        super().__init__(('127.0.0.1', 0), ReviewHandler)


class ReviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = self._route()
        if route is None:
            return
        app = self.server.app
        if route == 'state':
            self._send(json.dumps(app.snapshot(), allow_nan=False).encode(), 'application/json')
        elif route == 'frame':
            self._send(app.frame, 'image/jpeg')
        elif route in ('', 'app.js', 'style.css'):
            path = Path(__file__).with_name('web') / (route or 'index.html')
            self._send(path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'text/plain')
        elif route in ('Oswald.ttf', 'MerriweatherSans.ttf'):
            path = Path(__file__).parent / 'assets' / 'fonts' / route
            self._send(path.read_bytes(), 'font/ttf')
        else:
            self.send_error(404)

    def do_POST(self):
        route = self._route()
        if route is None:
            return
        app = self.server.app
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 0:
                raise ValueError('Invalid content length')
            if route == 'upload':
                if app.worker is not None:
                    raise ValueError('The review has already started')
                filename = Path(unquote(self.headers.get('X-Filename', 'video.mp4'))).name
                suffix = Path(filename).suffix.lower()
                if suffix not in ('.mp4', '.mov', '.avi', '.mkv', '.webm'):
                    raise ValueError('Choose a supported video file')
                directory = Path(app.uploads.name) / uuid.uuid4().hex
                directory.mkdir()
                path = directory / filename
                with path.open('wb') as output:
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError('Incomplete video upload')
                        output.write(chunk)
                        remaining -= len(chunk)
                self._send(json.dumps({'source': str(path)}).encode(), 'application/json')
                return
            if length > 16384:
                raise ValueError('Request is too large')
            values = json.loads(self.rfile.read(length) or b'{}')
            if route == 'action':
                app.action(values['action'])
            elif route == 'calibration':
                app.calibrate(values)
            elif route == 'start':
                app.start(values)
            elif route == 'quit':
                app.action('quit')
            else:
                self.send_error(404)
                return
            self._send(b'{}', 'application/json')
        except (ValueError, KeyError, TypeError, OSError) as error:
            self._send(json.dumps({'error': str(error)}).encode(), 'application/json', 400)

    def _route(self):
        # The unguessable path prevents unrelated websites from controlling the runner.
        expected_host = f'127.0.0.1:{self.server.server_port}'
        if self.headers.get('Host') != expected_host:
            self.send_error(403)
            return None
        origin = self.headers.get('Origin')
        if origin and origin != f'http://{expected_host}':
            self.send_error(403)
            return None
        prefix = f'/{self.server.app.token}/'
        path = urlsplit(self.path).path
        if not path.startswith(prefix):
            self.send_error(404)
            return None
        return path[len(prefix):]

    def _send(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass
