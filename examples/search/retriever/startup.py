"""Register, start and verify the local retrieval dependency before training."""
import argparse
import json
import os
from pathlib import Path
import pwd
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

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


def register(owner_pid, gpu, url):
    instructions = (ROOT / 'AGENTS.md').read_text()
    real_name = re.search(r'`real_name`:\s*([^\n]+)', instructions).group(1).strip()
    name_id = re.search(r'`name_id`:\s*`([^`]+)`', instructions).group(1)
    uid, started = lifecycle.process_identity(owner_pid)
    if uid != os.getuid():
        raise RuntimeError('retrieval owner must belong to this Linux user')
    data = ROOT.parent.parent / 'datasets' / 'searchR1'
    model = ROOT.parent.parent / 'weights' / 'e5-base-v2'
    # These match the asset paths used by retrieval_launch.sh.
    for path in (data/'e5_Flat.index', data/'wiki-18.jsonl', model):
        if not path.exists():
            raise FileNotFoundError(path)
    snapshot = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,memory.total',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    rows = {int(row.split(',')[0]): row.split(',') for row in snapshot.splitlines()}
    if gpu not in rows or int(rows[gpu][1]) < 2048:
        raise RuntimeError(f'retrieval GPU {gpu} needs at least 2 GiB free')
    mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(mem['MemAvailable'].split()[0]) / 1024**2
    if available < 80:
        raise RuntimeError('retrieval startup requires approximately 80 GiB available RAM')
    job = f'{name_id}-retrieval-{uuid.uuid4().hex[:12]}'
    log = ROOT / f'retrieval_{job}.log'
    lifecycle.append_event(job, 'REGISTERED', registered_at=lifecycle.now(), real_name=real_name,
        name_id=name_id, linux_user=pwd.getpwuid(uid).pw_name, host=socket.gethostname(), ai='codex-generated-launcher',
        task_name='Training-owned SearchR1 retrieval service', purpose='Search dependency; ready and warm before training',
        workdir=str(ROOT), entry_file=str(ROOT/'examples/search/retriever/retrieval_server.py'),
        configuration_files=[str(ROOT/'examples/search/retriever/retrieval_launch.sh')],
        training_files=[str(ROOT/'examples/search/retriever/retrieval_server.py'), str(Path(__file__).resolve())],
        dataset=str(data), model=str(model),
        command_redacted=f'CUDA_VISIBLE_DEVICES={gpu} RETRIEVAL_OWNER_PID={owner_pid} RETRIEVAL_JOB_ID={job} bash examples/search/retriever/retrieval_launch.sh --port {urllib.parse.urlsplit(url).port}',
        gpu_request={'ids':[gpu], 'count':1, 'vram_estimate_gb':'1-2'},
        cpu_cores_estimate='1-152 (FAISS defaults)', ram_estimate_gb='65-85', disk_growth_estimate_gb='<0.1 logs',
        duration_estimate='Startup about 5-10 minutes; service lasts until training exits',
        estimate_basis='61 GiB index; full index 278s and corpus 218s on prior run; GPU encoder about 1 GiB',
        log_path=str(log), checkpoint_path=None, result_path=str(log),
        resource_snapshot={'gpu_memory_csv':snapshot.strip(),'ram_available_gib':round(available,1)},
        owner_identity={'pid':owner_pid,'uid':uid,'start_time':started})
    record = lifecycle.latest_event(job)
    if (record['real_name'],record['name_id'],record['status']) != (real_name,name_id,'REGISTERED'):
        raise RuntimeError('registered identity readback failed')
    print(f'[retrieval] 登记完成：{real_name} ({name_id}); job_id={job}; GPU {gpu}, VRAM 1-2 GB, '
          f'CPU 1-152 cores, RAM 65-85 GB; startup 5-10 min, then training lifetime; '
          f'registry={lifecycle.REGISTRY}; log={log}', flush=True)
    return job, log


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
    event = lifecycle.latest_event(state['job_id'])
    if event['event'] not in ('SUCCEEDED','FAILED','CANCELLED','EXPIRED'):
        raise RuntimeError('retrieval stopped without a terminal registration event')
    path.unlink()
    print('[retrieval] Owned service stopped; terminal registration verified.', flush=True)


def start(args):
    url = urllib.parse.urlsplit(args.url)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1','localhost') or not url.port or url.path != '/retrieve':
        raise ValueError('managed search requires http://127.0.0.1:PORT/retrieve')
    with socket.socket() as sock:
        sock.settimeout(.3)
        if sock.connect_ex(('127.0.0.1',url.port)) == 0:
            raise RuntimeError(f'port {url.port} is already occupied; refusing to adopt or kill an unrelated service')
    job, log = register(args.owner_pid, args.gpu, args.url)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), RETRIEVAL_OWNER_PID=str(args.owner_pid),
               RETRIEVAL_JOB_ID=job, RETRIEVAL_STOP_OWNER_ON_FAILURE='1')
    # This supervisor closes its own registration before exiting; the generic
    # trainer cleanup must not kill it halfway through that operation.
    env.pop('AGENTOPSD_RUN_ID',None)
    env['PYTHON_BIN'] = os.environ.get('RETRIEVAL_PYTHON','/home/shuixia/miniconda3/envs/llm_ft_py310/bin/python')
    child = None
    try:
        with log.open('w') as stream:
            child = subprocess.Popen(['bash',str(ROOT/'examples/search/retriever/retrieval_launch.sh'),
                                      '--port',str(url.port)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        state = {'job_id':job,'supervisor_pid':child.pid,
                 'supervisor_identity':lifecycle.process_identity(child.pid),'log_path':str(log)}
        path=Path(args.state_file)
        with path.open('x') as stream:
            os.chmod(path,0o600);json.dump(state,stream)
        def service_pid():
            event=lifecycle.latest_event(job)
            return event.get('worker_pids',[None])[0] if event['event']=='RUNNING' else None
        wait_ready(args.url,child,service_pid,timeout=args.timeout)
        print(f'[retrieval] READY: {args.url}; real search passed. Training may start.',flush=True)
    except BaseException:
        if child is not None:
            child.terminate() if child.poll() is None else None
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        event=lifecycle.latest_event(job)
        if event['event']=='REGISTERED':
            lifecycle.append_event(job,'FAILED',ended_at=lifecycle.now(),exit_code=1,
                                   reason='Retrieval supervisor failed before binding',result_path=str(log))
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
