"""Real MCP transports with a controllable Docker boundary and failure injection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from src.review_backend import AgentReviewError, run_fresh_agent_json
from src.research.agent_execution_tools import verify_execution_tools
from src.research.agent_execution_tools import PreparedExecutionTools
from src.research.agent_source_tools import verify_source_tools
from src.research.agent_tool_profile import prepare_tool_profile
from src.research.config import load_config
from src.research.finite_python import FinitePythonBroker, _save
from src.research.harness import _author_settings, _reviewer_settings
from src.research.novelty_review import InvalidReviewResume, draft_research_agenda, review_novelty
from tests.research.test_agent_source_tools import environment, small_packet
from tests.research.test_novelty_gate import study as _study_fixture
from tests.research.test_unified_research import make_task

IMAGE = "sha256:" + "a" * 64

DOCKER = r'''
import json,sys,time,uuid
from pathlib import Path
root=Path(__ROOT__)
state=root/'containers.json'
items=json.loads(state.read_text())
mode=(root/'mode').read_text()
a=sys.argv[1:]
def save(): state.write_text(json.dumps(items))
def one(flag): return a[a.index(flag)+1]
if a[:2]==['image','inspect']:
    print(a[2]);sys.exit(0)
if a[0]=='create':
    cid=uuid.uuid4().hex+uuid.uuid4().hex
    labels=dict(a[i+1].split('=',1) for i,x in enumerate(a) if x=='--label')
    mount=dict(x.split('=',1) if '=' in x else (x,True) for x in one('--mount').split(','))
    h={'ReadonlyRootfs':'--read-only' in a,'NetworkMode':one('--network'),
       'CapDrop':[one('--cap-drop')],'Privileged':False,'DeviceRequests':None,'Devices':[],
       'Binds':None,'SecurityOpt':[one('--security-opt')],'Memory':536870912,'MemorySwap':536870912,
       'NanoCpus':int(float(one('--cpus'))*1e9),'PidsLimit':int(one('--pids-limit'))}
    if mode=='wrong-network':h['NetworkMode']='host'
    if mode=='wrong-owner':labels['agsci.finite.session']='unrelated-owner'
    items[cid]={'Id':cid,'Name':'/'+one('--name'),'Image':a[a.index('-I')-1],
       'Config':{'User':one('--user'),'Labels':labels},'HostConfig':h,
       'Mounts':[{'Type':'bind','Source':mount['src'],'Destination':mount['dst'],'RW':False}],
       'State':{'Running':False,'ExitCode':0,'OOMKilled':False}}
    save();print(cid);sys.exit(0)
identity=a[-1]
found=next((k for k,v in items.items() if k==identity or v['Name']=='/'+identity),None)
if found is None:
    print('Error: No such object: '+identity,file=sys.stderr);sys.exit(1)
if a[0]=='inspect':print(json.dumps([items[found]]));sys.exit(0)
if a[0]=='rm':
    if mode=='cleanup-fails':sys.exit(1)
    del items[found];save();print(identity);sys.exit(0)
if a[0]=='start':
    items[found]['State']['Running']=True;save()
    if mode=='sleep':time.sleep(120)
    if mode=='overflow':
        print('x'*200000,flush=True);time.sleep(120)
    rc=7 if mode=='program-error' else 0
    items[found]['State'].update(Running=False,ExitCode=rc);save()
    print('failure' if rc else '{"value":45}',flush=True)
    sys.exit(rc)
raise AssertionError(a)
'''

NATIVE = r'''
import json,os,subprocess,sys,tomllib,uuid
from pathlib import Path
assert sys.argv[sys.argv.index('--allow')+1]=='read-only'
home=Path(os.environ['CODEX_HOME'])
profile=tomllib.loads((home/(sys.argv[sys.argv.index('--profile')+1]+'.config.toml')).read_text())
observed={}
for name,spec in profile['mcp_servers'].items():
    proc=subprocess.Popen([spec['command'],*spec['args']],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    def rpc(i,method,params=None):
        proc.stdin.write(json.dumps({'jsonrpc':'2.0','id':i,'method':method,'params':params or {}})+'\n');proc.stdin.flush()
        r=json.loads(proc.stdout.readline());assert 'error' not in r,r;return r['result']
    def tool(i,name,args):
        result=rpc(i,'tools/call',{'name':name,'arguments':args})
        if name!='run_python':assert not result['isError'],result
        return json.loads(result['content'][0]['text'])
    rpc(1,'initialize',{'protocolVersion':'2025-06-18','capabilities':{},'clientInfo':{'name':'fixture','version':'1'}})
    rpc(2,'tools/list')
    if name=='agsci_sources':
        sources=tool(3,'list_sources',{})
        tool(4,'read_source_span',{'source_id':sources['sources'][0]['id'],'start':0,'end':3})
    else:
        assert spec['enabled_tools']==['get_execution_policy','run_python']
        assert spec['tools']['run_python']['approval_mode']=='approve'
        tool(3,'get_execution_policy',{})
        observed=tool(4,'run_python',{'code':'import json\nprint(json.dumps({"value":sum(INPUT_DATA["values"])}))','input':{'values':list(range(10))}})
    proc.stdin.close();proc.wait(timeout=10);assert proc.returncode==0,proc.stderr.read()
result=json.loads(__MESSAGE__) if __MESSAGE__ else {'execution':observed,'codex_home':str(home),'private_profile':str(home/(sys.argv[sys.argv.index('--profile')+1]+'.config.toml'))}
prompt=Path(sys.argv[sys.argv.index('--prompt-file')+1]).read_text()
if __REFLECT__ and 'PREVIOUS AGENDA JSON' in prompt:result['_reflection']={'done':True,'changes':'','remaining_gaps':[]}
print(json.dumps({'protocolVersion':1,'type':'result','exitCode':0,'data':{'finalMessage':json.dumps(result),'nativeSessionId':str(uuid.uuid4()),'model':sys.argv[sys.argv.index('--model')+1]}}))
'''


@pytest.fixture
def docker_boundary(tmp_path, monkeypatch):
    runtime = tmp_path/'docker-boundary'
    runtime.mkdir()
    (runtime/'containers.json').write_text('{}')
    (runtime/'mode').write_text('normal')
    executable = runtime/'docker'
    executable.write_text(f'#!{sys.executable}\n'+DOCKER.replace('__ROOT__', repr(str(runtime))))
    executable.chmod(0o700)
    monkeypatch.setenv('PATH', str(runtime)+os.pathsep+os.environ['PATH'])
    # Short, private aliases also exercise the existing daemon-path validator.
    with tempfile.TemporaryDirectory(prefix='fp-', dir=str(Path('/tmp').resolve())) as directory:
        root = Path(directory)/'b'
        monkeypatch.setenv('AGENTIC_BROKER_ROOT', str(root))
        monkeypatch.setenv('AGENTIC_BROKER_HOST_ROOT', str(root))
        yield runtime


def native_environment(path, message=None, *, reflect=False):
    env = environment(path, message, reflect=reflect)
    body = NATIVE.replace('__MESSAGE__', repr(json.dumps(message) if message is not None else ''))
    body = body.replace('__REFLECT__', repr(reflect))
    Path(env['HEADLESS_BIN']).write_text(f'#!{sys.executable}\n'+body)
    return env


def invoke(tmp_path, *, packet=True, timeout=15):
    env = native_environment(tmp_path)
    original = {p.name: p.read_bytes() for p in Path(env['CODEX_HOME']).iterdir()}
    output = tmp_path/'transport'
    payload, receipt = run_fresh_agent_json(
        'Verify the finite calculation.', work_dir=output, model='codex/fixture', profile='fixture',
        environment=env, source_lookup_packet=small_packet() if packet else None,
        python_execution_image=IMAGE, timeout_seconds=timeout,
    )
    assert original == {p.name: p.read_bytes() for p in Path(env['CODEX_HOME']).iterdir()}
    return output, payload, receipt


def test_composed_tools_and_execution_only_use_private_profiles_and_bound_receipts(tmp_path, docker_boundary):
    for include_sources in [True, False]:
        path = tmp_path/str(include_sources)
        path.mkdir()
        output, payload, receipt = invoke(path, packet=include_sources)
        binding = verify_execution_tools(output, receipt['python_execution'], expected_prompt_sha256=receipt['prompt_sha256'])
        assert binding['execution_jobs_verified'] == 1 and binding['tool_catalog_delivered']
        assert payload['execution']['stdout'] == '{"value":45}\n'
        assert not Path(payload['codex_home']).exists()
        if include_sources:
            sources = verify_source_tools(output, receipt['source_lookup'], expected_packet=small_packet())
            assert sources['private_profile_sha256'] == binding['private_profile_sha256']
        assert not list(output.rglob('auth.json'))
        assert not list(output.rglob('*token*'))
        assert not list(Path(os.environ['AGENTIC_BROKER_ROOT']).glob('finite-python-*/session.json'))
    assert json.loads((docker_boundary/'containers.json').read_text()) == {}


@pytest.mark.parametrize('filename', ['program.py', 'input.json', 'stdout.bin', 'stderr.bin', 'receipt.json', 'isolation.json', 'calls.jsonl', 'runner.py', 'bridge.py', 'adapter.py', 'binding.json'])
def test_execution_evidence_tampering_rejected(tmp_path, docker_boundary, filename):
    output, _, receipt = invoke(tmp_path)
    root = output/'python_execution'
    path = root/filename if (root/filename).exists() else next(root.glob('agsci-python-*'))/filename
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError):
        verify_execution_tools(output, receipt['python_execution'], expected_prompt_sha256=receipt['prompt_sha256'])


@pytest.mark.parametrize('mode,expected', [('program-error', 'program_failed'), ('wrong-network', 'infrastructure_error')])
def test_program_errors_and_isolation_failures_are_not_promoted(tmp_path, docker_boundary, mode, expected):
    (docker_boundary/'mode').write_text(mode)
    output, payload, receipt = invoke(tmp_path)
    assert payload['execution']['status'] == expected
    assert payload['execution']['cleanup_verified']
    assert json.loads((docker_boundary/'containers.json').read_text()) == {}
    verify_execution_tools(output, receipt['python_execution'], expected_prompt_sha256=receipt['prompt_sha256'])


def test_cancelled_native_call_stops_running_program_and_service(tmp_path, docker_boundary):
    (docker_boundary/'mode').write_text('sleep')
    with pytest.raises(AgentReviewError, match='timed out'):
        invoke(tmp_path, timeout=2)
    assert json.loads((docker_boundary/'containers.json').read_text()) == {}
    assert not list(Path(os.environ['AGENTIC_BROKER_ROOT']).glob('finite-python-*/session.json'))
    assert json.loads((tmp_path/'transport/receipt.json').read_text())['status']=='failed'


def test_cleanup_failure_retains_recovery_and_prevents_completion(tmp_path, docker_boundary):
    (docker_boundary/'mode').write_text('cleanup-fails')
    with pytest.raises(AgentReviewError, match='cleanup'):
        invoke(tmp_path)
    roots = list(Path(os.environ['AGENTIC_BROKER_ROOT']).glob('finite-python-*'))
    assert len(roots) == 1 and (roots[0]/'session.json').exists()
    output = tmp_path/'transport/python_execution'
    archived = tmp_path/'preserved-previous-execution'
    output.rename(archived)
    before = {p.relative_to(archived).as_posix(): p.read_bytes() for p in archived.rglob('*') if p.is_file()}
    (docker_boundary/'mode').write_text('normal')
    broker = FinitePythonBroker(output, image=IMAGE, prompt_sha256='b'*64)
    assert broker.recovered_jobs == 1
    broker.close()
    assert json.loads((docker_boundary/'containers.json').read_text()) == {}
    assert before == {p.relative_to(archived).as_posix(): p.read_bytes() for p in archived.rglob('*') if p.is_file()}


def test_active_lease_and_unacknowledged_creation_are_fail_closed(tmp_path, docker_boundary):
    artifacts = tmp_path/'jobs'
    broker = FinitePythonBroker(artifacts, image=IMAGE, prompt_sha256='b'*64)
    with pytest.raises(BlockingIOError):
        FinitePythonBroker(artifacts, image=IMAGE, prompt_sha256='b'*64)
    broker.state['jobs'].append({'name':'agsci-python-'+'c'*32,'cid':None,'cleaned':False})
    _save(broker.state_path, broker.state)
    with pytest.raises(ValueError, match='not acknowledged'):
        broker.close()
    assert broker.state_path.exists()
    artifacts.rename(tmp_path/'preserved')
    with pytest.raises(ValueError, match='not acknowledged'):
        FinitePythonBroker(artifacts, image=IMAGE, prompt_sha256='b'*64)


def test_ownership_mismatch_preserves_container_and_recovery_record(tmp_path, docker_boundary):
    (docker_boundary/'mode').write_text('wrong-owner')
    broker=FinitePythonBroker(tmp_path/'jobs',image=IMAGE,prompt_sha256='b'*64)
    result=broker.execute({'code':'print(1)'})
    assert result['status']=='cleanup_unverified' and not result['cleanup_verified']
    assert len(json.loads((docker_boundary/'containers.json').read_text()))==1
    with pytest.raises(ValueError,match='ownership differs'):
        broker.close()
    assert broker.state_path.exists()


def test_http_auth_and_origin_rejected_without_program_execution(tmp_path,docker_boundary):
    env=native_environment(tmp_path)
    temporary=tmp_path/'private';temporary.mkdir(mode=0o700)
    profile=prepare_tool_profile(temporary=temporary,environment=env,agent='codex',model='codex/fixture',profile='fixture')
    artifacts=tmp_path/'artifacts';artifacts.mkdir()
    prepared=PreparedExecutionTools(artifacts=artifacts,temporary=temporary,profile=profile,image=IMAGE,prompt_sha256='b'*64)
    url=f'http://127.0.0.1:{prepared.server.server_port}/mcp'
    token=prepared.token_path.read_text()
    def post(payload,headers=None):
        request=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json',**(headers or {})})
        with urllib.request.urlopen(request,timeout=5) as response:return json.loads(response.read())
    try:
        with pytest.raises(urllib.error.HTTPError) as error:post({'jsonrpc':'2.0','id':1,'method':'initialize'})
        assert error.value.code==401
        with pytest.raises(urllib.error.HTTPError) as error:
            post({'jsonrpc':'2.0','id':1,'method':'initialize'},{'Authorization':'Bearer '+token,'Origin':'http://attacker.invalid'})
        assert error.value.code==403
        for i,method in enumerate(['initialize','tools/list']):
            post({'jsonrpc':'2.0','id':i,'method':method},{'Authorization':'Bearer '+token})
        receipt=prepared.finish()
        assert verify_execution_tools(artifacts,receipt,expected_prompt_sha256='b'*64)['execution_jobs_verified']==0
        assert prepared.broker.calls==0 and not prepared.token_path.exists()
    finally:prepared.close()


def test_author_reflection_and_review_cache_verify_execution_before_new_model_call(tmp_path, docker_boundary, monkeypatch):
    task, cfg, agenda, review = _study_fixture.__wrapped__(tmp_path)
    author_env = tmp_path/'author-env';author_env.mkdir()
    settings = {'agent':'codex','model':'codex/fixture','profile':'fixture','timeout_seconds':15,
                'environment':native_environment(author_env, agenda, reflect=True),'source_lookup':True,
                'python_execution_image':IMAGE}
    authors = tmp_path/'authors';authors.mkdir()
    result = draft_research_agenda(task, cfg, authors, reviewer_settings=settings, num_reflections=2)
    assert result == agenda
    reviewer_env = tmp_path/'reviewer-env';reviewer_env.mkdir()
    review_settings = {**settings,'environment':native_environment(reviewer_env, review)}
    reviews = tmp_path/'reviews'
    assert review_novelty(task, agenda, cfg, reviews, reviewer_settings=review_settings)['status']=='supported'
    for directory in [authors/'agenda_draft',authors/'agenda_reflection_2',reviews/'reviewer_1',reviews/'reviewer_2']:
        receipt = json.loads((directory/'receipt.json').read_text())
        assert receipt['python_execution_requested'] is True
        verify_execution_tools(directory/'transport',receipt['python_execution'],expected_prompt_sha256=receipt['prompt_sha256'])
    monkeypatch.setattr('src.review_backend.run_fresh_agent_json', lambda *a,**kw: pytest.fail('Completed native context rerolled'))
    assert draft_research_agenda(task,cfg,authors,reviewer_settings=settings,num_reflections=2,resume=True)==agenda
    assert review_novelty(task,agenda,cfg,reviews,reviewer_settings=review_settings,resume=True)['status']=='supported'
    path=next((reviews/'reviewer_2/transport/python_execution').glob('agsci-python-*'))/'stdout.bin'
    path.write_text('altered')
    with pytest.raises(InvalidReviewResume):
        review_novelty(task,agenda,cfg,reviews,reviewer_settings=review_settings,resume=True)


def test_harness_image_opt_in_and_explicit_native_backend(tmp_path):
    cfg=SimpleNamespace(agent=SimpleNamespace(name='codex',model='author',profile='author',reasoning_effort='low'),
        review=SimpleNamespace(backend='agent',agent_name=None,agent_model='reviewer',agent_profile='reviewer',agent_reasoning_effort='high',timeout_seconds=60))
    assert 'python_execution_image' not in _author_settings(cfg)
    assert _author_settings(cfg,python_execution_image=IMAGE)['model']=='author'
    assert _reviewer_settings(cfg,python_execution_image=IMAGE)['model']=='reviewer'
    cfg.review.backend='api'
    with pytest.raises(ValueError,match='native Codex'):
        _reviewer_settings(cfg,python_execution_image=IMAGE)
    task=make_task(tmp_path/'task')
    text=(task/'research.yaml').read_text()
    (task/'research.yaml').write_text(text+'\nharness:\n  agent_python_image: '+IMAGE+'\n')
    assert load_config(task).harness.agent_python_image==IMAGE
    (task/'research.yaml').write_text(text+'\nharness:\n  agent_python_image: latest\n')
    with pytest.raises(ValueError,match='immutable'):
        load_config(task)
