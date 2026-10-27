"""Inspect returned endpoint files before separate OS/transfer provenance audit.

No model execution, checkpoint selection, or declaration of scientific success.
"""
import hashlib,json,math,pathlib
from dataclasses import asdict
from src.p6_schedule import TimingCell


def bind_training(root, directory, expected_cell, expected_job_sha256, *, stage):
    if stage not in ("prefix", "tail"):
        raise ValueError("Unsupported training stage")
    start,end=(1,3000) if stage=="prefix" else (3001,4000)
    root=pathlib.Path(root).resolve();folder=(root/directory).resolve()
    if pathlib.Path(directory).is_absolute() or not folder.is_relative_to(root):
        raise ValueError('Endpoint outside project')
    cell=TimingCell(**expected_cell)
    if stage=="prefix" and 3001 in (cell.aux_start,cell.mechanism_start,cell.data_start):
        raise ValueError("Prefix must use its canonical early-only cell")
    paths={name:folder/name for name in ('latest.pt','bn-buffers.pt','summary.json','train.jsonl')}
    for p in paths.values():
        if not p.resolve().is_relative_to(root) or not p.is_file():
            raise ValueError('Missing or out-of-project endpoint artifact')
    hashes={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in paths.items()}
    summary=json.loads(paths['summary.json'].read_text())
    expected={'status':'complete','stage':stage,'cell':asdict(cell),'start_step':start,
              'end_step':end,'updates':end-start+1,'job_sha256':expected_job_sha256,'test_access':False,'bn_recalibration_batches':128,
              'checkpoint_sha256':hashes['latest.pt'],'bn_sha256':hashes['bn-buffers.pt']}
    if not expected_job_sha256 or any(summary.get(k)!=v for k,v in expected.items()):
        raise ValueError('Endpoint summary identity or SHA mismatch')
    trace=[json.loads(line) for line in paths['train.jsonl'].read_text().splitlines()]
    if [x.get('step') for x in trace]!=list(range(start,end+1)):
        raise ValueError('Incomplete, duplicate or out-of-order training trace')
    for row in trace:
        schedule=cell.at(row['step'])
        arm=schedule['mechanism'] or ('B-same-branch-data' if schedule['branch_data'] else 'B')
        if row.get('idm')!=schedule['idm'] or row.get('active_arm')!=arm:
            raise ValueError('Trace does not match the planned timing cell')
        if any(isinstance(v,(int,float)) and not math.isfinite(v) for v in row.values()):
            raise ValueError('Nonfinite training trace')
    return {'status':'files_checked_pending_os_and_transfer_audit',
            ('evaluation_input' if stage=='tail' else 'parent_input'):{'cell':asdict(cell),'source_kind':'p6_'+stage,
                'path':str(paths['latest.pt'].relative_to(root)),
                'bn_path':str(paths['bn-buffers.pt'].relative_to(root)),
                'checkpoint_sha256':hashes['latest.pt'],'bn_sha256':hashes['bn-buffers.pt'],
                'job_sha256':expected_job_sha256},
            'artifacts':[{'path':str(paths[n].relative_to(root)),'sha256':h} for n,h in hashes.items()],
            'launch_authorized':False,'checkpoint_payload_validated':False}


def bind_endpoint(root, directory, expected_cell, expected_job_sha256):
    return bind_training(root,directory,expected_cell,expected_job_sha256,stage='tail')


def bind_prefix(root, directory, expected_cell, expected_job_sha256):
    result=bind_training(root,directory,expected_cell,expected_job_sha256,stage='prefix')
    result['parent_input']['prefix_key']=list(TimingCell(**expected_cell).prefix_key())
    return result
