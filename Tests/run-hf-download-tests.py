#!/usr/bin/env python3
"""Exercise actual local HTTP interruption/resume/cancellation, no model downloads."""
import argparse, hashlib, json, socket, subprocess, threading, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
p.add_argument('--derived-data', type=Path, required=True)
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
products = args.derived_data / 'Build/Products/Release'
commit = 'a' * 40
config = b'{"model_type":"fixture"}'
weight = b'0123456789abcdef' * 16384
state = {'weight_gets': 0, 'range': 0, 'config_gets': 0, 'heads': 0, 'redirect_gets': 0}
def git_hash(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_HEAD(self):
        state['heads'] += 1
        if self.headers.get('Authorization') != 'Bearer fixture-key':
            self.send_response(403); self.end_headers(); return
        self.send_response(302)
        self.send_header('Location', f'http://127.0.0.1:{self.server.server_port}/must-not-follow')
        oid = 'b'*40 if '/bad-hash/' in self.path else hashlib.sha256(weight).hexdigest()
        self.send_header('X-Linked-Etag', f'"{oid}"')
        self.send_header('X-Linked-Size', str(len(weight) + (1 if '/bad-size/' in self.path else 0)))
        self.send_header('X-Repo-Commit', 'b'*40 if '/bad-commit/' in self.path else commit)
        self.end_headers()
    def do_GET(self):
        path = self.path.split('?')[0]
        if '/resolve/' in path:
            if self.headers.get('Authorization') != 'Bearer fixture-key':
                self.send_response(401); self.end_headers(); return
        if path.startswith('/api/models/fixture/') and '/revision/' in path:
            data = json.dumps({'id': 'fixture/'+path.split('/')[4], 'sha': commit}).encode()
        elif path.startswith('/api/models/fixture/') and '/tree/' in path:
            lfs = {'size':len(weight), 'pointerSize':128}
            if '/missing-hash/' not in path: lfs['oid'] = '*'*64
            data = json.dumps([
                {'type':'file', 'path':'config.json', 'size':len(config), 'oid':git_hash(config)},
                {'type':'file', 'path':'model.safetensors', 'size':len(weight), 'oid':'b'*40,
                 'lfs':lfs}
            ]).encode()
        elif path == '/must-not-follow':
            state['redirect_gets'] += 1; data = b'not a metadata response'
        elif path == '/stats': data = json.dumps(state, sort_keys=True).encode()
        elif path.endswith('/config.json'):
            state['config_gets'] += 1; data = config
        elif path.endswith('/model.safetensors'):
            state['weight_gets'] += 1
            offset = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
            if offset: state['range'] = offset
            self.send_response(206 if offset else 200)
            self.send_header('Content-Length', str(len(weight)-offset))
            if offset: self.send_header('Content-Range', f'bytes {offset}-{len(weight)-1}/{len(weight)}')
            self.end_headers()
            if state['weight_gets'] == 1:
                self.wfile.write(weight[:65536]); self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR); self.connection.close(); return
            self.wfile.write(weight[offset:]); return
        elif path in ['/ignore','/slow']:
            data = b'A'*4096
            self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers()
            if path == '/slow':
                self.wfile.write(data[:1024]); self.wfile.flush(); time.sleep(2)
                try: self.wfile.write(data[1024:])
                except (BrokenPipeError, ConnectionResetError): pass
            else: self.wfile.write(data)
            return
        else: self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)

server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
exe = root / '.build/HFModelDownloadTests'
sdk = subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
sources = ['Tests/HFModelDownloadTests.swift','Applications/Synomancer/Services/HFModelDownloadService.swift',
           'Applications/Synomancer/Services/HFStreamingDownload.swift','Applications/Synomancer/Services/SpeechCacheAccess.swift']
cmd = ['swiftc','-swift-version','6','-parse-as-library','-O','-sdk',sdk,'-target','arm64-apple-macos27.0',
       '-module-cache-path',str(root/'.build/TestModuleCache'),'-I',str(products)]
cmd += [str(root/s) for s in sources]
cmd += [str(products/(name+'.o')) for name in ['HuggingFace','Crypto','Logging','EventSource']]
cmd += ['-o',str(exe)]
subprocess.run(cmd,check=True,cwd=root)
import tempfile
with tempfile.TemporaryDirectory(dir=root/'.build',prefix='HFDownloadTests-') as cache:
    subprocess.run([str(exe),f'http://127.0.0.1:{server.server_port}',cache],check=True)
server.shutdown()
