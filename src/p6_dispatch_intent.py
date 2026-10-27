"""Durable, exclusive dispatch intent. Never starts a process or clears a marker.

Call only after live prerequisites pass and immediately before controller launch.
Any existing marker, including a torn write, requires process/lease inspection;
this module deliberately offers no retry or delete operation.
"""
import json
import os
from pathlib import Path

from src.p6_dispatch import bound_dispatch_arguments
from scripts.audit_p6_endpoint_inputs import read_bound
from src.p6_receipt_audit import PROJECT


def claim_dispatch_intent(root, spec, *, now):
    root=Path(root).resolve()
    bound=bound_dispatch_arguments(root,spec,now=now)
    job=read_bound(root,spec['job'])
    directory=root/'receipts/p6-dispatch-intents'
    directory.mkdir(parents=True,exist_ok=True)
    if not directory.resolve().is_relative_to(root):
        raise ValueError('Intent directory outside project')
    path=directory/(job['reservation_id']+'.json')
    record={'project_id':PROJECT,'job_id':job['run_name'],'reservation_id':job['reservation_id'],
            'phase':'intent_recorded','created_at':now.isoformat(),'job':spec['job'],
            'bound_configuration':bound,'launch_performed':False,
            'meaning':'Submission may become unknown after this record; never infer absence of a process from this file.'}
    # O_EXCL is the concurrency guard. Do not remove a partial file on error.
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
    with os.fdopen(fd,'w') as out:
        json.dump(record,out,ensure_ascii=False,indent=2)
        out.write('\n');out.flush();os.fsync(out.fileno())
    directory_fd=os.open(directory,os.O_RDONLY)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)
    return {'intent_path':str(path.relative_to(root)),'record':record,'launch_performed':False}
