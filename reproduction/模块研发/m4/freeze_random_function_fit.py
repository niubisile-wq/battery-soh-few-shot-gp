"""Freeze all105 source-fit variants; no best-initialization selection."""
import json
from pathlib import Path
from source_oof import sa, HERE
from batch_random_function_fit import checked
from freeze_random_function_private_controls import build_manifest as controls_manifest


def build_manifest(results):
    batch = results / 'm4_random_function_fit_batch_v1'
    result = json.loads((batch / 'result.json').read_text()); req = json.loads((batch / 'request.json').read_text())
    assert result['status'] == 'ALL_SOURCE_REPLAY_AUDITED' and len(result['folds']) == 21
    assert {r['fold'] for r in result['folds']} == set(req['folds']) and len(set(req['folds'])) == 21
    assert all(sa.digest(HERE / n) == h for n, h in req['code_hashes'].items())
    protocol = json.loads((HERE / 'random_function_fit_stability_protocol.json').read_text()); assert protocol == req['protocol']
    source = results / 'm4_random_function_private_controls_v1'
    frozen = json.loads((source / 'frozen_selections.json').read_text()); assert frozen == controls_manifest(results)
    audit = json.loads((source / 'verification.json').read_text())
    assert audit['status'] == 'PASS' and audit['summary_sha256'] == sa.digest(source / 'summary.json')
    originals = {s['fold']:s['choices']['function_private'] for s in frozen['selections']}
    selections = []; model_flags = []
    for entry in sorted(result['folds'], key=lambda r:r['fold']):
        root = Path(entry['root']); fold = entry['fold']; assert entry == checked(root, fold, protocol)
        fit_req = json.loads((root / 'request.json').read_text()); r = json.loads((root / 'result.json').read_text())
        assert fit_req['choice'] == originals[fold] and r['original_replay_error'] < 1e-8
        choices = {}
        for model in r['models']:
            multiplier = model['multiplier']; family = 'init_' + str(multiplier)
            choices[family] = dict(selected=dict(target=family, key=originals[fold]['selected']['key'],
                gain=originals[fold]['selected']['gain'], initialization_multiplier=multiplier, selection_rule='Fixed original private0.5; no new validation selection'), model=model)
            model_flags.append(dict(fold=fold, multiplier=multiplier, optimization=model['optimization'],
                diagnostics=model['diagnostics'], kernel_bound_coordinates=model['kernel_bound_coordinates'],
                initialization=model['initialization'], source_nll=model['source_nll']))
        selections.append(dict(fold=fold, choices=choices, fit_root=str(root), fit_result_sha256=sa.digest(root / 'result.json'),
            fit_audit_sha256=sa.digest(root / 'verification.json')))
    assert len(model_flags) == 105
    return dict(status='ALL_INITIALIZATIONS_FROZEN_NO_QUERY_SCORES', primary='init_1.0', selections=selections, model_flags=model_flags,
        protocol=protocol, parent_manifest_sha256=frozen['parent_manifest_sha256'], m2_manifest_sha256=frozen['m2_manifest_sha256'],
        batch_result_sha256=sa.digest(batch / 'result.json'), batch_request_sha256=sa.digest(batch / 'request.json'),
        original_frozen_sha256=sa.digest(source / 'frozen_selections.json'), original_audit_sha256=sa.digest(source / 'verification.json'),
        original_summary_sha256=sa.digest(source / 'summary.json'), code_sha256=sa.digest(Path(__file__)),
        limits='All105 fits retained, no best-run selection. Source numerical initialization sensitivity only, not whole-pipeline random retraining or independent data.')


def main():
    results = HERE.parent / 'results'; manifest = build_manifest(results)
    out = results / 'm4_random_function_fit_screen_v1'; out.mkdir(exist_ok=False)
    sa.write_json(out / 'frozen_selections.json', manifest)
    print('21folds x5source initializations frozen', flush=True)


if __name__ == '__main__': main()
