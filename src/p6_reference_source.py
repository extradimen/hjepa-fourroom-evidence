"""Revalidate the SHA-bound P5 task reference without deserializing tensors."""
import hashlib
from pathlib import Path
from scripts.audit_p6_endpoint_inputs import read_bound
from src.p6_receipt_audit import PROJECT


def validate_reference_source(root, reference, *, tasks, recipe):
    source = read_bound(root, reference)
    if source.get('project_id') != PROJECT or source.get('status') != 'existing_p5_reference_source_bound':
        raise ValueError('Missing P5 reference source binding')
    if source.get('reference_tasks') != tasks:
        raise ValueError('Reference tasks differ from source binding')
    refs = source['evidence_references']
    if refs['recipe'] != recipe:
        raise ValueError('Reference recipe differs from P5 source')
    evidence = {k: read_bound(root, v) for k, v in refs.items()}
    task_path=(Path(root)/tasks['path']).resolve()
    if Path(tasks['path']).is_absolute() or not task_path.is_relative_to(Path(root).resolve()):
        raise ValueError('Reference task outside project')
    if hashlib.sha256(task_path.read_bytes()).hexdigest()!=tasks['sha256']:
        raise ValueError('Reference task SHA mismatch')
    manifest = evidence['return_manifest']
    matches = [m for m in manifest['materials'] if m['destination'] == tasks['path']]
    if len(matches) != 1 or matches[0]['sha256'] != tasks['sha256'] or matches[0]['source'] != source['source']:
        raise ValueError('Reference absent from return manifest')
    material = matches[0]
    composite, prior = evidence['completed_composite'], evidence['prior_transfer']
    manifest_sha = refs['return_manifest']['sha256']
    if (composite.get('status') != 'complete' or composite.get('manifest_sha256') != manifest_sha
            or composite.get('prior_receipt') != refs['prior_transfer']['path']
            or material not in composite.get('prior_verified_materials', [])):
        raise ValueError('Reference recovery completion mismatch')
    if prior.get('project_id') != PROJECT or prior.get('manifest_sha256') != manifest_sha:
        raise ValueError('Reference prior transfer identity mismatch')
    events = [e for e in prior['materials'] if e.get('destination') == tasks['path']]
    if len(events) != 1:
        raise ValueError('Reference successful transfer event missing or ambiguous')
    event = events[0]
    expected = dict(event='materialized', project_id=PROJECT, node_id='local-mac',
                    sha256=tasks['sha256'], source=material['source'], manifest_sha256=manifest_sha)
    if any(event.get(k) != v for k, v in expected.items()) or event != source['successful_material_event']:
        raise ValueError('Reference successful transfer event mismatch')
    summary, audit, node = evidence['summary'], evidence['p5_final_audit'], evidence['node_state']
    if (summary.get('status') != 'complete' or summary.get('test_access') is not False
            or summary.get('task_count') != 32 or summary.get('task_sha256') != tasks['sha256']
            or summary.get('recipe_sha256') != recipe['sha256']):
        raise ValueError('Reference summary mismatch')
    matches = [r for r in audit.get('results', []) if all(r.get(k) == summary.get(k) for k in ('seed', 'idm', 'arm'))]
    if audit.get('status') != 'returned_p5_evidence_audited' or audit.get('task_payloads_exactly_equal_to_p4') is not True or len(matches) != 1:
        raise ValueError('Reference lacks P5 final acceptance')
    if node.get('termination_evidence') != 'os_child_wait_status' or node.get('return_code') != 0 or node['child_process_now'].get('alive') is not False:
        raise ValueError('Reference source process exit unverified')
    if material['source'] != {'kind':'project', 'node':node['node_id'],
                             'path':f"runs/p5-planning-v1/seed{summary['seed']}-idm{summary['idm']}-{summary['arm']}/tasks.pt"}:
        raise ValueError('Reference task source run mismatch')
    return {'status':'p5_reference_source_revalidated', 'reference':reference,
            'model_reuse_authorized':False}
