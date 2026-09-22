import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from examples.search.retriever import lifecycle


def test_wait_ready_requires_matching_service_and_warmup():
    from examples.search.retriever.startup import wait_ready
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            calls.append('health')
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({'status':'ready', 'pid':os.getpid()}).encode())
        def do_POST(self):
            calls.append('warmup')
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({'result':[[{'document':{'contents':'ok'}}]]}).encode())
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    class Child:
        def poll(self): return None
    try:
        url=f'http://127.0.0.1:{server.server_port}/retrieve'
        wait_ready(url,Child(),lambda:os.getpid(),timeout=2,interval=.01)
        assert calls == ['health','warmup']
        with pytest.raises(RuntimeError,match='different process'):
            wait_ready(url,Child(),lambda:os.getpid()+1,timeout=.1,interval=.01)
        assert calls[-1]=='health'
    finally:
        server.shutdown();server.server_close();thread.join()


def test_exited_service_fails_before_warmup():
    from examples.search.retriever.startup import wait_ready
    class Child:
        def poll(self): return 1
    with pytest.raises(RuntimeError,match='exited'):
        wait_ready('http://127.0.0.1:1/retrieve',Child(),lambda:1,timeout=1,interval=.01)


def test_missing_service_has_bounded_startup_wait():
    from examples.search.retriever.startup import wait_ready
    class Child:
        def poll(self): return None
    with pytest.raises(TimeoutError,match='ready'):
        wait_ready('http://127.0.0.1:1/retrieve',Child(),lambda:1,timeout=.05,interval=.01)


def test_search_launchers_start_service_before_training():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for size in ['3b','7b']:
        source=(root/f'examples/agentopsd_trainer/run_search_{size}.sh').read_text()
        assert source.index('agentopsd_search_setup') < source.index('-m verl.trainer.main_opsd')
        assert 'env.search.search_url="$SEARCH_URL"' in source


@pytest.mark.parametrize('startup_code', [0, 1])
def test_actual_launcher_never_trains_before_service_ready(tmp_path, startup_code):
    import subprocess
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    events=tmp_path/'events'
    fake=tmp_path/'python'
    fake.write_text(f'''#!{sys.executable}
import os,sys,pathlib
args=sys.argv[1:]
events=pathlib.Path(os.environ['EVENTS'])
if args[0].endswith('startup.py'):
    action=args[1]
    with events.open('a') as f:f.write(action+'\\n')
    if action=='start':raise SystemExit({startup_code})
elif args[:2]==['-m','verl.trainer.main_opsd']:
    assert events.read_text().splitlines()==['start']
    with events.open('a') as f:f.write('train\\n')
''')
    fake.chmod(0o755)
    env={**os.environ,'PYTHON_BIN':str(fake),'EVENTS':str(events),'RETRIEVAL_CHECK_ONLY':'0'}
    result=subprocess.run(['bash',str(root/'examples/agentopsd_trainer/run_search_3b.sh')],
                          cwd=root,env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==startup_code, result.stderr
    assert events.read_text().splitlines()==(['start','train','stop'] if startup_code==0 else ['start','stop'])
