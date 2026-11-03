"""Check real normalization source; --device cuda exercises compiled kernels."""
import argparse
import ast
import importlib
from pathlib import Path
import sys
import types
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--device',default='cpu')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
for name,path in [('trellis2','trellis2'),('trellis2.modules','trellis2/modules'),
                  ('trellis2.modules.attention','trellis2/modules/attention'),
                  ('trellis2.modules.sparse','trellis2/modules/sparse'),
                  ('trellis2.modules.sparse.attention','trellis2/modules/sparse/attention')]:
    package=types.ModuleType(name)
    package.__path__=[str(root/path)]
    sys.modules[name]=package
basic=importlib.import_module('trellis2.modules.sparse.basic')
sparse=sys.modules['trellis2.modules.sparse']
sparse.VarLenTensor=basic.VarLenTensor
sparse.SparseTensor=basic.SparseTensor
# Load the real LayerNorm32/manual_cast definitions without importing convolution
# backends from modules.utils; no normalization implementation is substituted.
def definition(path,name):
    return next(node for node in ast.parse((root/path).read_text()).body
                if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name==name)
norms=types.ModuleType('trellis2.modules.norm')
norms.__package__='trellis2.modules'
norms.__dict__.update(torch=torch,nn=torch.nn)
sys.modules[norms.__name__]=norms
nodes=[definition('trellis2/modules/utils.py','manual_cast'),definition('trellis2/modules/norm.py','LayerNorm32')]
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'norm_source','exec'),norms.__dict__)
dense=importlib.import_module('trellis2.modules.attention.modules')
sparse_modules=importlib.import_module('trellis2.modules.sparse.attention.modules')
fast=importlib.import_module('trellis2.modules.fast_norm')
model=torch.nn.Module()
block=torch.nn.Module()
block.norm1=norms.LayerNorm32(128,elementwise_affine=False,eps=1e-6)
block.norm2=norms.LayerNorm32(128,elementwise_affine=True,eps=1e-6)
block.self_attn=dense.MultiHeadRMSNorm(64,2)
block.cross_attn=sparse_modules.SparseMultiHeadRMSNorm(64,2)
model.blocks=torch.nn.ModuleList([block])
model.eval().to(args.device)
torch.manual_seed(42)
dtype=torch.float16 if args.device=='cuda' else torch.float32
matrix=torch.randn(23,128,device=args.device,dtype=dtype)
heads=matrix.reshape(23,2,64)
packed=basic.VarLenTensor.from_tensor_list([heads[:7],heads[7:]])
weights={key:value.clone() for key,value in model.state_dict().items()}
def evaluate():
    return [block.norm1(matrix),block.norm2(matrix),block.self_attn(heads),block.cross_attn(packed).feats]
with torch.no_grad():
    reference=evaluate()
    assert fast.set_compiled_norms(model,True)==4
    if args.device=='cuda':
        actual=evaluate()
        for output,expected in zip(actual,reference):
            assert torch.isfinite(output).all() and output.dtype==expected.dtype
            torch.testing.assert_close(output,expected,atol=.003,rtol=.003)
    else:
        try:
            evaluate()
        except ValueError as error:
            assert 'CUDA' in str(error)
        else:
            raise AssertionError('Compiled normalization must reject CPU input')
    fast.set_compiled_norms(model,False)
    assert all(torch.equal(a,b) for a,b in zip(evaluate(),reference))
assert all(torch.equal(value,model.state_dict()[key]) for key,value in weights.items())
print(f'PASS: {args.device} LayerNorm/head norms, ragged layout, disable parity, weights and CPU guard')
