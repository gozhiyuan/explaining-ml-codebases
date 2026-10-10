"""CPU contracts of original AST methods, with explicit stand-ins.
Run: python tests/source_contracts.py REPO_ROOT
Not an end-to-end vLLM run; no model, tensors or real scheduler instance.
"""
import ast
import copy
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace as NS

REPO = Path(sys.argv[1])
COMMIT = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
assert COMMIT == 'd29f7f5c9294be8e489dac34d45a939b95a06336', 'use the pinned evaluation checkout'
records = []
def load(file, cls, symbol, env=None):
    path = REPO / file
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = copy.deepcopy(next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == symbol))
    node.decorator_list = []
    scope = {'__builtins__': __builtins__}
    scope.update(env or {})
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), scope)
    records.append({'file': file, 'symbol': cls+'.'+symbol, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return scope[symbol]

single='vllm/v1/core/single_type_kv_cache_manager.py'
manager_file='vllm/v1/core/kv_cache_manager.py'
class Spec:
    block_size=16
find=load(single,'FullAttentionManager','find_longest_cache_hit',{'FullAttentionSpec':Spec,'ChunkedLocalAttentionSpec':Spec,'resolve_block_hashes':lambda hashes,*a,**k:hashes,'itertools':itertools,'cdiv':lambda n,d:(n+d-1)//d})
class Pool:
    hash_block_size=16
    def __init__(self, present, free=10):
        self.present=present; self.free=free
    def get_cached_block(self,h,g):
        return [NS(block_id=self.present[h],ref_cnt=0)] if h in self.present else None
    def get_num_free_blocks(self): return self.free
cls=NS(supports_fine_grained_hash_lookup=True)
# Prompt 35: two 16-token prefix blocks fit below the last-token cap.
blocks,hit=find(cls,['h1','h2'],34,[0],Pool({'h1':1,'h2':2}),Spec(),False,16)
assert hit==32 and [b.block_id for b in blocks[0]]==[1,2]
_,miss=find(cls,['different','h2'],34,[0],Pool({'h1':1,'h2':2}),Spec(),False,16)
assert miss==0
# Exact 32-token prompt: last-token cap 31 allows only one full block.
_,capped=find(cls,['h1','h2'],31,[0],Pool({'h1':1,'h2':2}),Spec(),False,16)
assert capped==16

get=load(manager_file,'KVCacheManager','get_computed_blocks')
probe=NS(empty_kv_cache_blocks=([],),prefix_cache_lookup_enabled=lambda r:True,
    coordinator=NS(find_longest_cache_hit=lambda h,maxlen:(blocks,32,0)),
    enable_kv_cache_events=False,create_kv_cache_blocks=lambda x:x)
assert get(probe,NS(num_tokens=35,block_hashes=['h1','h2']))==(blocks,32,0)
probe.prefix_cache_lookup_enabled=lambda r:False
assert get(probe,NS(num_tokens=35,block_hashes=['h1','h2']))==(([],),0,0)

count=load(single,'SingleTypeKVCacheManager','get_num_blocks_to_allocate',{'cdiv':lambda n,d:(n+d-1)//d})
state=NS(block_size=16,_max_admission_blocks_per_request=None,req_to_blocks={},num_cached_block={},get_num_skipped_tokens=lambda n:0,_get_num_evictable_blocks=lambda b:sum(x.ref_cnt==0 for x in b),_has_partial_local_hit=lambda b,n:False)
assert count(state,'R',35,blocks[0],32,32,35)==3 # 1 new + 2 idle hits removed from free queue
state.req_to_blocks['R']=[1,2,3]; state.num_cached_block['R']=2
assert count(state,'R',36,[],35,35,36)==0
assert count(state,'R',49,[],48,48,49)==1

allocate=load(manager_file,'KVCacheManager','allocate_slots',{'RequestStatus':NS(WAITING='WAITING',PREEMPTED='PREEMPTED')})
# Fake coordinator records side-effect order; arithmetic/control flow is original.
events=[]
coordinator=NS(remove_skipped_blocks=lambda *a,**k:events.append('remove_skipped'),get_num_blocks_to_allocate=lambda **k:3,
    allocate_new_computed_blocks=lambda **k:events.append('attach_prefix'),
    allocate_new_blocks=lambda *a:events.append('allocate_new') or ([NS(block_id=3)],),
    cache_blocks=lambda r,n:events.append(('register_cache',n)))
a=NS(empty_kv_cache_blocks=NS(blocks=([],)),max_model_len=128,watermark_blocks=0,coordinator=coordinator,block_pool=Pool({},free=10),enable_caching=True,create_kv_cache_blocks=lambda b:b)
r=NS(request_id='R',status='WAITING',num_tokens=35,num_prompt_tokens=35,num_computed_tokens=0,num_in_flight_tokens=0)
result=allocate(a,r,3,num_new_computed_tokens=32,new_computed_blocks=NS(blocks=blocks),has_scheduled_reqs=False)
assert result[0][0].block_id==3
assert events==['remove_skipped','attach_prefix','allocate_new',('register_cache',35)]
events.clear(); a.block_pool.free=2
assert allocate(a,r,3,num_new_computed_tokens=32,new_computed_blocks=NS(blocks=blocks),has_scheduled_reqs=False) is None
assert events==['remove_skipped'] # no attachment/allocation/index registration after capacity failure

# Original scheduler arithmetic statements, selected by AST; excludes outer queues,
# encoder/spec/Mamba branches and does not validate the whole schedule method.
scheduler='vllm/v1/core/sched/scheduler.py'
tree=ast.parse((REPO/scheduler).read_text())
owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
schedule=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='schedule')
loop=next(n for n in schedule.body if isinstance(n,ast.While))
i=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='num_new_tokens' for t in n.targets))
snippet=ast.Module(body=copy.deepcopy(loop.body[i:i+4]),type_ignores=[])
for computed,known,expected in [(0,35,16),(16,35,16),(32,35,3),(35,36,1),(36,37,1)]:
    env={'self':NS(scheduler_config=NS(long_prefill_token_threshold=0),max_model_len=128,num_sampled_tokens_per_step=1), 'request':NS(num_computed_tokens=computed,num_tokens_with_spec=known,num_output_placeholders=0),'token_budget':16,'input_budget':16,'draft_slots':0}
    exec(compile(ast.fix_missing_locations(copy.deepcopy(snippet)),scheduler,'exec'),env)
    assert env['num_new_tokens']==expected
records.append({'file':scheduler,'symbol':'Scheduler.schedule selected running-token arithmetic statements','sha256':hashlib.sha256((REPO/scheduler).read_bytes()).hexdigest()})
report={'status':'pass','contracts':8,'contract_groups':['whole-prefix hit/miss/last-token cap','manager normal/skipped lookup','idle-hit free demand','running block growth','successful allocation order','capacity-failure order','cold running arithmetic','decode running arithmetic'],'environment':{'python':platform.python_version(),'os':platform.system(),'architecture':platform.machine(),'device':'CPU only; no tensor/model execution'},'results_boundary':'Lookup/capacity/allocation execute original AST methods with stand-ins. Schedule sequences execute four original running-arithmetic statements; waiting branches are source-reviewed.','boundary':'Original AST methods and four arithmetic statements; fake coordinator/pool/spec; no whole Scheduler, GPU, tokenizer, model or real KV writes.', 'results':{'warm_prefix_hit':hit,'changed_first_block_hit':miss,'exact_32_prompt_hit':capped,'miss_schedule':[16,16,3,1,1],'hit_schedule':[3,1,1],'idle_hit_capacity_required':3,'cache_registration_before_execution':True},'source_records':records}
Path(__file__).with_name('source-contract-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
