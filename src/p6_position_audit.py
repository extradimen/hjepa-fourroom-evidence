"""Recompute diagnostic metrics from returned coordinates, without model execution."""
import math,statistics
from src.p6_schedule import TimingCell


def position_metrics(summary,*,inputs,recipe,recipe_sha256,job_sha256):
    expected={'status':'complete','recipe_sha256':recipe_sha256,'job_sha256':job_sha256,
              'world_model_updates':0,'test_access':False,'closed_loop_evaluated':False,'privileged_diagnostic':True}
    if any(summary.get(k)!=v for k,v in expected.items()):raise ValueError('Position summary identity mismatch')
    if recipe.get('status')!='frozen' or recipe.get('test_access') is not False:raise ValueError('Unfrozen position recipe')
    lookup={TimingCell(**x['cell']):x for x in inputs}
    if len(lookup)!=len(inputs) or not inputs:raise ValueError('Duplicate or empty input cells')
    seen=set()
    def xy(value):
        if len(value)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in value):raise ValueError('Invalid coordinate')
        return value
    for row in summary['results']:
        cell=TimingCell(**row['cell']);variant=row['variant'];key=(cell,variant)
        if cell not in lookup or variant not in ('raw','bn_recalibrated') or key in seen:raise ValueError('Unknown or duplicate probe row')
        seen.add(key);item=lookup[cell]
        fields={'training_job_sha256':item['job_sha256'],'checkpoint_sha256':item['checkpoint_sha256'],
                'bn_sha256':item['bn_sha256'] if variant=='bn_recalibrated' else None,
                'train_samples':128*len(recipe['train_starts']),'dev_samples':32*len(recipe['dev_starts'])}
        if any(row.get(k)!=v for k,v in fields.items()):raise ValueError('Probe identity or sample count mismatch')
        clips=row['per_clip']
        if [(c['episode'],c['start']) for c in clips]!=[(e,t) for e in range(32) for t in recipe['dev_starts']]:raise ValueError('Probe clip order or coverage mismatch')
        mean=xy(row['train_mean_xy'])
        for metric,field in [('true_encoding','decoded_xy'),('predicted_encoding','forecast_xy'),('persistence','persistence_xy'),('train_mean',None)]:
            squared=[]
            for clip in clips:
                truth=xy(clip['true_xy']);estimate=xy(clip[field]) if field else mean
                squared.append(sum((a-b)**2 for a,b in zip(estimate,truth)))
            distances=[math.sqrt(v) for v in squared]
            calculated={'mean_distance_pixels':statistics.mean(distances),'median_distance_pixels':statistics.median(distances),
                        'rmse_per_coordinate_pixels':math.sqrt(sum(squared)/(2*len(squared)))}
            if any(not math.isclose(row[metric][k],v,rel_tol=1e-9,abs_tol=1e-7) for k,v in calculated.items()):raise ValueError('Probe metric arithmetic mismatch')
    if len(seen)!=2*len(inputs):raise ValueError('Missing raw or calibrated probe row')
    return {'status':'coordinate_metrics_checked_pending_provenance','rows':len(seen),
            'remaining':['OS and budget child waits','governed return SHA coverage','train-only fit and dataset coordinate provenance']}


def audit_position_return(root,spec):
    from scripts.audit_p6_endpoint_inputs import read_bound
    from src.p6_evaluation_identity import validate_training_binding
    from src.p6_receipt_audit import audit_evaluation_process,audit_return_coverage
    job,state,run_id=audit_evaluation_process(root,spec,stage='position')
    if job['p5_recipe_sha256']!=spec['recipe']['sha256'] or job['training_audit_path']!=spec['training_audit']['path'] or job['training_audit_sha256']!=spec['training_audit']['sha256']:
        raise ValueError('Position recipe or training audit binding mismatch')
    training=read_bound(root,spec['training_audit'])
    for item in job['inputs']:validate_training_binding(training,item)
    summary=read_bound(root,spec['summary']);recipe=read_bound(root,spec['recipe'])
    result=position_metrics(summary,inputs=job['inputs'],recipe=recipe,
        recipe_sha256=spec['recipe']['sha256'],job_sha256=spec['job']['sha256'])
    audit_return_coverage(read_bound(root,spec['return_manifest']),read_bound(root,spec['return_receipt']),
        manifest_sha256=spec['return_manifest']['sha256'],node_id=state['node_id'],run_name=job['run_name'],
        artifacts=[spec['summary']],source_relative_paths={spec['summary']['path']:'summary.json'})
    return {'status':'position_return_audited_pending_training_and_data_provenance','rows':result['rows'],
        'run_id':run_id,'launch_authorized':False,'evidence_references':{k:spec[k] for k in
        ('job','recipe','training_audit','summary','expected_launch','launch','node_state','reservation','budget_result','return_manifest','return_receipt')},
        'remaining':['training checkpoint payload and ancestry validation','train-only fit and dataset coordinate provenance']}
