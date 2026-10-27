"""Extract the executed P5 update without running its top-level CLI."""
import ast
import copy
import hashlib
import io


def tensor_difference_report(left, right, limit=12):
    """Report numerical differences without changing exact acceptance criteria."""
    import torch
    rows=[];total=0
    for name in sorted(left.keys() | right.keys()):
        a,b=left.get(name),right.get(name)
        if a is None and b is None:continue
        if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
            if a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b):continue
        total+=1
        if len(rows)>=limit:continue
        row={'name':name,'left_present':a is not None,'right_present':b is not None}
        if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
            row.update(left_shape=list(a.shape),right_shape=list(b.shape),left_dtype=str(a.dtype),right_dtype=str(b.dtype))
            if a.shape==b.shape:
                a,b=a.detach().cpu().double(),b.detach().cpu().double()
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                row.update(finite=finite,different_elements=int((a!=b).sum()),elements=a.numel())
                if finite and a.numel():
                    row.update(max_absolute_difference=float((a-b).abs().max()),reference_max_absolute=float(a.abs().max()))
        rows.append(row)
    return {'exact_equal':total==0,'mismatched_tensors':total,'details':rows,'details_truncated':total>len(rows)}


def assert_disjoint_tensor_storage(left, right):
    """Independent comparison paths must not share even CPU Adam step tensors."""
    import torch
    def storages(value):
        if isinstance(value, torch.Tensor):
            return {(str(value.device), value.untyped_storage().data_ptr())} if value.numel() else set()
        values = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else ()
        return set().union(*(storages(v) for v in values))
    if storages(left) & storages(right):
        raise ValueError('Parity paths share mutable tensor storage')


def diagnostic_roundtrip(model, head, optimizer):
    """Exercise tensor serialization without claiming a trained branch boundary."""
    import torch
    from src.p6_checkpoint import parameter_groups, validate_parent_state
    payload={'model':model.state_dict(), 'head':None if head is None else head.state_dict(),
             'optimizer':optimizer.state_dict(),
             'optimizer_parameter_names':parameter_groups(model,head,optimizer)}
    stream=io.BytesIO();torch.save(payload,stream);stream.seek(0)
    recovered=torch.load(stream,weights_only=False)
    validate_parent_state(recovered)
    exact_state_equal(payload,recovered)
    if parameter_groups(model,head,optimizer)!=recovered['optimizer_parameter_names']:
        raise ValueError('Diagnostic optimizer parameter order mismatch')
    model.load_state_dict(recovered['model'])
    if head is not None:head.load_state_dict(recovered['head'])
    optimizer.load_state_dict(recovered['optimizer'])


def p5_update_kernel(source, expected_sha256, namespace):
    if hashlib.sha256(source).hexdigest()!=expected_sha256:
        raise ValueError('P5 source SHA mismatch')
    tree=ast.parse(source)
    loops=[n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step']
    if len(loops)!=1:raise ValueError('Ambiguous P5 training loop')
    body=loops[0].body
    starts=[i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='base' for t in n.targets)]
    ends=[i for i,n in enumerate(body) if isinstance(n,ast.Expr) and ast.unparse(n.value)=='r.optim.step()']
    if len(starts)!=1 or len(ends)!=1 or starts[0]>=ends[0]:raise ValueError('P5 update boundaries changed')
    class Arguments(ast.NodeTransformer):
        def visit_Attribute(self,node):
            if isinstance(node.value,ast.Name) and node.value.id=='args' and node.attr in ('arm','seed'):
                return ast.copy_location(ast.Name(id=node.attr,ctx=ast.Load()),node)
            return self.generic_visit(node)
    statements=[Arguments().visit(copy.deepcopy(n)) for n in body[starts[0]:ends[0]+1]]
    function=ast.parse('def original_update(r,b,o,head,reference,coef,arm,seed,step):\n pass').body[0]
    function.body=statements
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    scope=dict(namespace);exec(compile(module,'<SHA-bound-P5-update>','exec'),scope)
    return scope['original_update']


def exact_state_equal(left,right,path='state'):
    import torch
    if isinstance(left,torch.Tensor):
        if not isinstance(right,torch.Tensor) or left.dtype!=right.dtype or left.shape!=right.shape or not torch.equal(left,right):
            raise ValueError('Parity tensor mismatch: '+path)
    elif isinstance(left,dict):
        if not isinstance(right,dict) or left.keys()!=right.keys():raise ValueError('Parity keys mismatch: '+path)
        for key in left:exact_state_equal(left[key],right[key],path+'.'+str(key))
    elif isinstance(left,(list,tuple)):
        if type(left)!=type(right) or len(left)!=len(right):raise ValueError('Parity sequence mismatch: '+path)
        for i,(a,b) in enumerate(zip(left,right)):exact_state_equal(a,b,path+'.'+str(i))
    elif type(left)!=type(right) or left!=right:raise ValueError('Parity scalar mismatch: '+path)
