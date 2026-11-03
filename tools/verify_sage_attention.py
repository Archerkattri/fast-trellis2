"""Verify opt-in dispatch/packing on CPU, or real Sage kernels with --device cuda."""
import argparse
import ast
import copy
import importlib
from pathlib import Path
import sys
import types
import torch
import torch.nn.functional as F

parser=argparse.ArgumentParser()
parser.add_argument('--device',default='cpu')
parser.add_argument('--sage-package',type=Path)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
if args.sage_package:
    sys.path.insert(0,str(args.sage_package))
for name,path in [('trellis2','trellis2'),('trellis2.modules','trellis2/modules'),
                  ('trellis2.modules.attention','trellis2/modules/attention'),
                  ('trellis2.modules.sparse','trellis2/modules/sparse'),
                  ('trellis2.modules.sparse.attention','trellis2/modules/sparse/attention')]:
    package=types.ModuleType(name)
    package.__path__=[str(root/path)]
    sys.modules[name]=package
basic=importlib.import_module('trellis2.modules.sparse.basic')
sys.modules['trellis2.modules.sparse'].VarLenTensor=basic.VarLenTensor
dense=importlib.import_module('trellis2.modules.attention.full_attn')
dense.config.BACKEND='sdpa'
sparse=importlib.import_module('trellis2.modules.sparse.attention.full_attn')
sage=importlib.import_module('trellis2.modules.attention.sage')
real_sage=sage.sage_attention
calls=[]
def reference(q,k,v):
    return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)
def stub(q,k,v,backend):
    calls.append((backend,q.shape,k.shape))
    return reference(q,k,v)
if args.device=='cpu':
    sage.sage_attention=stub
dtype=torch.float32 if args.device=='cpu' else torch.float16
torch.manual_seed(42)
q,k,v=[torch.randn(2,8,2,64,device=args.device,dtype=dtype) for _ in range(3)]
expected=reference(q,k,v)
assert torch.equal(dense.scaled_dot_product_attention(q,k,v),expected)
tolerance=1e-5 if args.device=='cpu' else .08
for backend in ['sage','sage_fp16pv']:
    for values in [(q,k,v),(q,torch.stack([k,v],dim=2)),(torch.stack([q,k,v],dim=2),)]:
        actual=dense.scaled_dot_product_attention(*values,backend=backend)
        torch.testing.assert_close(actual,expected,atol=tolerance,rtol=tolerance)
    # Distinct per-sample constant values catch accidental cross-batch mixing.
    queries=[q[0,:3],q[1,:5]]
    keys=[k[0,:7],k[1,:2]]
    values=[torch.full_like(keys[0],7),torch.full_like(keys[1],-3)]
    Q,K,V=[basic.VarLenTensor.from_tensor_list(parts) for parts in [queries,keys,values]]
    out=sparse.sparse_scaled_dot_product_attention(Q,K,V,backend=backend)
    torch.testing.assert_close(out.feats[:3],torch.full_like(out.feats[:3],7),atol=tolerance,rtol=tolerance)
    torch.testing.assert_close(out.feats[3:],torch.full_like(out.feats[3:],-3),atol=tolerance,rtol=tolerance)
    parts=[torch.stack([queries[i],queries[i],torch.full_like(queries[i],value)],dim=1)
           for i,value in enumerate([7,-3])]
    packed=basic.VarLenTensor.from_tensor_list(parts)
    out=sparse.sparse_scaled_dot_product_attention(packed,backend=backend)
    torch.testing.assert_close(out.feats[:3],torch.full_like(out.feats[:3],7),atol=tolerance,rtol=tolerance)
    torch.testing.assert_close(out.feats[3:],torch.full_like(out.feats[3:],-3),atol=tolerance,rtol=tolerance)
    KV=basic.VarLenTensor.from_tensor_list([torch.stack([keys[i],values[i]],dim=1) for i in range(2)])
    out=sparse.sparse_scaled_dot_product_attention(Q,KV,backend=backend)
    torch.testing.assert_close(out.feats,V.feats[:0].new_tensor([7]*3+[-3]*5)[:,None,None].expand_as(out.feats),atol=tolerance,rtol=tolerance)

modules=importlib.import_module('trellis2.modules.attention.modules')
model=torch.nn.Module()
block=torch.nn.Module()
block.self_attn=modules.MultiHeadAttention(128,2,type='self')
block.cross_attn=modules.MultiHeadAttention(128,2,ctx_channels=128,type='cross')
model.blocks=torch.nn.ModuleList([block])
weights={key:value.clone() for key,value in model.state_dict().items()}
assert sage.set_sage_self_attention(model)==1
assert block.self_attn.attn_backend=='sage' and not hasattr(block.cross_attn,'attn_backend')
sage.set_sage_self_attention(model,None)
assert block.self_attn.attn_backend is None
assert all(torch.equal(value,model.state_dict()[key]) for key,value in weights.items())
sage.sage_attention=real_sage
if args.device=='cpu':
    tree=ast.parse((root/'trellis2/pipelines/trellis2_image_to_3d.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Trellis2ImageTo3DPipeline')
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='enable_sage_attention')
    scope={'__package__':'trellis2.pipelines'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'pipeline_api','exec'),scope)
    flows={f'{stage}_slat_flow_model_{resolution}':copy.deepcopy(model)
           for stage in ['shape','tex'] for resolution in [512,1024]}
    owner=types.SimpleNamespace(models=flows,default_pipeline_type='1024_cascade')
    configure=scope['enable_sage_attention']
    configure(owner,pipeline_type='512')
    configure(owner,pipeline_type='1024_cascade')
    for name,flow in flows.items():
        assert flow.blocks[0].self_attn.attn_backend==('sage' if name.endswith('1024') else None)
        assert not hasattr(flow.blocks[0].cross_attn,'attn_backend')
    configure(owner,None)
    assert all(flow.blocks[0].self_attn.attn_backend is None for flow in flows.values())
    print('PASS: pipeline selection, 512-to-cascade reconfiguration and disable')
print(f'PASS: {args.device} dense/ragged QKV dispatch, batch isolation, native default, model weights and cross-attention')
