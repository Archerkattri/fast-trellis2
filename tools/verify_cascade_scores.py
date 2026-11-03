"""Check SS score bypass without CUDA/model dependencies, using real method source."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch

path=Path(__file__).resolve().parents[1]/'trellis2/pipelines/trellis2_image_to_3d.py'
tree=ast.parse(path.read_text())
cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Trellis2ImageTo3DPipeline')
method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='sample_sparse_structure')
method.decorator_list=[]
method.returns=None
for arg in method.args.args:
    arg.annotation=None
calls=[]
def scores(value,**kwargs):
    calls.append(value)
    return torch.arange(int(value.sum())).float()
scope={'torch':torch,'get_coords_value':lambda x:x,'process_and_visualize':scores}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),scope)
occupancy=torch.full((1,1,4,4,4),-1.)
occupancy[0,0,1,1,1]=1
occupancy[0,0,2,2,2]=1
owner=SimpleNamespace(models={'sparse_structure_flow_model':SimpleNamespace(resolution=4,in_channels=1),
    'sparse_structure_decoder':lambda x:occupancy},low_vram=False,device='cpu',
    sparse_structure_sampler_params={},enable_faster=True,enable_voxel_visualization=False,
    sparse_structure_sampler=SimpleNamespace(sample=lambda *a,**kw:SimpleNamespace(samples=a[1])))
sample=scope['sample_sparse_structure']
torch.manual_seed(42)
reference,weights=sample(owner,{},4)
state=torch.get_rng_state()
assert len(calls)==1 and len(weights)==len(reference)==2
torch.manual_seed(42)
actual,weights=sample(owner,{},4,collect_scores=False)
assert weights is None and len(calls)==1
assert torch.equal(actual,reference) and torch.equal(torch.get_rng_state(),state)
print('PASS: identical occupancy coordinates and RNG state; unused scoring bypassed')

# Exercise the real pipeline dispatch across sequential generations. Models and
# final decoding are stubs; SS sampling and score lifecycle are real source.
run=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='run')
run.decorator_list=[]
run.returns=None
for arg in run.args.args:
    arg.annotation=None
exec(compile(ast.fix_missing_locations(ast.Module(body=[run],type_ignores=[])),str(path),'exec'),scope)
leader=SimpleNamespace(scores='stale')
leader.set_coords_scores=lambda value:setattr(leader,'scores',value)
owner.shape_slat_sampler=leader
owner.sample_sparse_structure=lambda *a,**kw:sample(owner,*a,**kw)
owner.get_cond=lambda *a:{}
owner.models.update({key:object() for key in ['shape_slat_flow_model_512',
    'shape_slat_flow_model_1024','tex_slat_flow_model_512','tex_slat_flow_model_1024']})
owner.sample_shape_slat=lambda *a,**kw:'shape'
def cascade(*args,**kwargs):
    assert leader.scores is None,'Previous run scores leaked into cascade'
    return 'shape',1024
owner.sample_shape_slat_cascade=cascade
owner.sample_tex_slat=lambda *a,**kw:'texture'
owner.decode_latent=lambda *a:['mesh']
pipeline_run=scope['run']
before=len(calls)
occupancy=torch.full((1,1,64,64,64),-1.)
occupancy[0,0,2,2,2]=1
occupancy[0,0,4,4,4]=1
pipeline_run(owner,None,preprocess_image=False,pipeline_type='512')
assert len(calls)==before+1 and leader.scores is not None
pipeline_run(owner,None,preprocess_image=False,pipeline_type='1024_cascade')
assert len(calls)==before+1 and leader.scores is None
owner.enable_voxel_visualization=True
pipeline_run(owner,None,preprocess_image=False,pipeline_type='1536_cascade')
assert len(calls)==before+2 and leader.scores is None
print('PASS: sequential 512/cascade score lifecycle and visualization opt-in')
