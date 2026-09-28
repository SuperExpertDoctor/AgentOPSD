"""Register, start and verify the local retrieval dependency before training."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from examples.search.retriever import lifecycle


def wait_ready(url, child, service_pid, *, timeout=1200, interval=2):
    """Do not release training until this exact service passes a real search."""
    parsed = urllib.parse.urlsplit(url)
    health = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, '/health', '', ''))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    next_report = time.monotonic() + 30
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError('retrieval service exited during startup; inspect its log')
        expected = service_pid()
        if expected is not None:
            try:
                with opener.open(health, timeout=min(2, max(.01, deadline-time.monotonic()))) as response:
                    data = json.load(response)
            except (urllib.error.URLError, TimeoutError, OSError, ValueError):
                data = None
            if data and data.get('status') == 'ready':
                if data.get('pid') != expected:
                    raise RuntimeError('retrieval port belongs to a different process')
                request = urllib.request.Request(url, data=json.dumps({
                    'query': 'What is the capital of France?', 'topk': 3, 'return_scores': True,
                }).encode(), headers={'Content-Type': 'application/json'})
                with opener.open(request, timeout=min(60, max(.01, deadline-time.monotonic()))) as response:
                    result = json.load(response)
                if not result.get('result') or not result['result'][0]:
                    raise RuntimeError('retrieval warmup returned no documents')
                if child.poll() is not None:
                    raise RuntimeError('retrieval service exited after warmup')
                return
        if time.monotonic() >= next_report:
            print('[retrieval] Loading index/model; training has not started.', flush=True)
            next_report = time.monotonic() + 30
        time.sleep(min(interval, max(0, deadline-time.monotonic())))
    raise TimeoutError('retrieval service did not become ready before startup deadline')


def retrieval_assets(data_dir, weights_dir):
    return Path(data_dir) / 'searchR1', Path(weights_dir) / 'e5-base-v2'


def check_resources(gpu, data_dir, weights_dir):
    data, model = retrieval_assets(data_dir, weights_dir)
    for path in (data / 'e5_Flat.index', data / 'wiki-18.jsonl', model):
        if not path.exists():
            raise FileNotFoundError(path)
    snapshot = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,memory.total',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    rows = {int(row.split(',')[0]): row.split(',') for row in snapshot.splitlines()}
    if gpu not in rows or int(rows[gpu][1]) < 2048:
        raise RuntimeError(f'retrieval GPU {gpu} needs at least 2 GiB free')
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(mem['MemAvailable'].split()[0]) / 1024**2
    if available < 80:
        raise RuntimeError('retrieval startup requires approximately 80 GiB available RAM')


def stop(state_path):
    path = Path(state_path)
    if not path.exists():
        return
    state = json.loads(path.read_text())
    pid = state['supervisor_pid']
    identity = tuple(state['supervisor_identity'])
    if lifecycle.owner_alive(pid, identity):
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic()+15
        while lifecycle.owner_alive(pid, identity) and time.monotonic()<deadline:
            time.sleep(.1)
        if lifecycle.owner_alive(pid, identity):
            raise RuntimeError(f'retrieval supervisor {pid} did not finish cleanup')
    Path(state['service_pid_file']).unlink(missing_ok=True)
    path.unlink()
    print('[retrieval] Owned service stopped.', flush=True)


def start(args):
    url = urllib.parse.urlsplit(args.url)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1','localhost') or not url.port or url.path != '/retrieve':
        raise ValueError('managed search requires http://127.0.0.1:PORT/retrieve')
    with socket.socket() as sock:
        sock.settimeout(.3)
        if sock.connect_ex(('127.0.0.1',url.port)) == 0:
            raise RuntimeError(f'port {url.port} is already occupied; refusing to adopt or kill an unrelated service')
    data_dir = os.environ.get('ASSET_DATA_DIR', '/root/autodl-fs/datasets')
    weights_dir = os.environ.get('ASSET_WEIGHTS_DIR', '/root/autodl-fs/cache/weights')
    check_resources(args.gpu, data_dir, weights_dir)
    owner_identity = lifecycle.process_identity(args.owner_pid)
    if owner_identity[0] != os.getuid():
        raise RuntimeError('retrieval owner must belong to this Linux user')
    pid_file = str(Path(args.state_file).with_name('service.json'))
    log = ROOT / f'retrieval_{args.owner_pid}_{Path(args.state_file).parent.name}.log'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), RETRIEVAL_OWNER_PID=str(args.owner_pid),
               RETRIEVAL_OWNER_START_TIME=str(owner_identity[1]),
               RETRIEVAL_SERVICE_PID_FILE=pid_file, RETRIEVAL_STOP_OWNER_ON_FAILURE='1',
               ASSET_DATA_DIR=data_dir, ASSET_WEIGHTS_DIR=weights_dir)
    # The supervisor owns cleanup of the service process group.
    env.pop('AGENTOPSD_RUN_ID',None)
    env['PYTHON_BIN'] = os.environ.get('RETRIEVAL_PYTHON', sys.executable)
    child = None
    try:
        with log.open('w') as stream:
            child = subprocess.Popen(['bash',str(ROOT/'examples/search/retriever/retrieval_launch.sh'),
                                      '--port',str(url.port)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        state = {'supervisor_pid':child.pid, 'service_pid_file':pid_file,
                 'supervisor_identity':lifecycle.process_identity(child.pid),'log_path':str(log)}
        path=Path(args.state_file)
        with path.open('x') as stream:
            os.chmod(path,0o600);json.dump(state,stream)
        def service_pid():
            try:
                service = json.loads(Path(pid_file).read_text())
                return (service['pid'] if lifecycle.process_identity(service['pid'])[1] == service['start_time']
                        else None)
            except (OSError, ValueError, KeyError):
                return None
        wait_ready(args.url,child,service_pid,timeout=args.timeout)
        print(f'[retrieval] READY: {args.url}; real search passed. Training may start.',flush=True)
    except BaseException:
        if child is not None:
            child.terminate() if child.poll() is None else None
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        Path(pid_file).unlink(missing_ok=True)
        Path(args.state_file).unlink(missing_ok=True)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','stop'])
    parser.add_argument('--state-file',required=True)
    parser.add_argument('--owner-pid',type=int)
    parser.add_argument('--url',default='http://127.0.0.1:8081/retrieve')
    parser.add_argument('--gpu',type=int,default=6)
    parser.add_argument('--timeout',type=float,default=1200)
    args=parser.parse_args()
    try:
        if args.action=='start':start(args)
        else:stop(args.state_file)
    except Exception as exc:
        print(f'[retrieval] Startup/cleanup failed: {exc}',file=sys.stderr,flush=True)
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
