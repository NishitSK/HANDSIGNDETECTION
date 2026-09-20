"""
Serve the mobile web app over HTTPS on the local network.

Phone browsers only expose the camera on secure origins, so plain HTTP on a LAN
address won't work. This serves mobile/app/ with a self-signed certificate that is
generated on first run (and regenerated if this PC's LAN IP changes). Your phone will
show a certificate warning that you have to accept once.

    python mobile/serve_https.py [--port 8443]

For the verification pages in mobile/tests/, serve the whole project over plain HTTP
on this PC only (browsers treat localhost as secure, so no certificate is needed):

    python mobile/serve_https.py --test [--port 8082]
"""
import argparse
import functools
import http.server
import shutil
import socket
import ssl
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_DIR = HERE / 'app'
PROJECT_DIR = HERE.parent
# Outside APP_DIR on purpose: anything under the served directory is downloadable.
CERT_DIR = HERE / '.certs'

# Windows can register .js as text/plain, which makes browsers refuse module scripts.
MIME_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.mjs': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json',
    '.webmanifest': 'application/manifest+json',
    '.bin': 'application/octet-stream',
    '.f32': 'application/octet-stream',
    '.wasm': 'application/wasm',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
}


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, **MIME_TYPES}

    def send_head(self):
        # --test serves the project root, which contains the certificate's private key.
        if '.certs' in Path(self.translate_path(self.path)).parts:
            self.send_error(404)
            return None
        return super().send_head()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


def lan_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            # UDP connect sends nothing; it just picks the outbound interface.
            s.connect(('10.255.255.255', 1))
            return s.getsockname()[0]
        except OSError:
            return '127.0.0.1'


def find_openssl():
    found = shutil.which('openssl')
    if found:
        return found
    for candidate in (r'C:\Program Files\Git\usr\bin\openssl.exe',
                      r'C:\Program Files\Git\mingw64\bin\openssl.exe'):
        if Path(candidate).exists():
            return candidate
    raise SystemExit('openssl not found. Install Git for Windows or add openssl to PATH.')


def ensure_certificate(ip):
    cert, key, ip_file = CERT_DIR / 'cert.pem', CERT_DIR / 'key.pem', CERT_DIR / 'ip.txt'
    if cert.exists() and key.exists() and ip_file.exists() and ip_file.read_text().strip() == ip:
        return cert, key

    CERT_DIR.mkdir(exist_ok=True)
    result = subprocess.run(
        [find_openssl(), 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
         '-keyout', str(key), '-out', str(cert), '-days', '365',
         '-subj', '/CN=ISL Mobile (local dev)',
         '-addext', f'subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost'],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f'Certificate generation failed:\n{result.stderr}')
    ip_file.write_text(ip)
    return cert, key


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int)
    parser.add_argument('--test', action='store_true',
                        help='serve the project folder over plain HTTP on localhost only, for mobile/tests/')
    args = parser.parse_args()

    if args.test:
        port = args.port or 8082
        server = http.server.ThreadingHTTPServer(
            ('127.0.0.1', port), functools.partial(Handler, directory=str(PROJECT_DIR)))
        print(f'Serving {PROJECT_DIR} on this PC only')
        print(f'  Model parity:      http://localhost:{port}/mobile/tests/parity.html')
        print(f'  MediaPipe parity:  http://localhost:{port}/mobile/tests/mediapipe_parity.html')
        print(f'  App:               http://localhost:{port}/mobile/app/')
    else:
        port = args.port or 8443
        ip = lan_ip()
        cert, key = ensure_certificate(ip)
        server = http.server.ThreadingHTTPServer(
            ('0.0.0.0', port), functools.partial(Handler, directory=str(APP_DIR)))
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        print(f'Serving {APP_DIR}')
        print(f'  On your phone (same Wi-Fi):  https://{ip}:{port}')
        print(f'  On this PC:                  https://localhost:{port}')
        print('Accept the certificate warning on first visit. If the phone cannot connect,')
        print('allow Python through Windows Firewall for private networks.')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
