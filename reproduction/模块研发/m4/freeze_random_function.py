"""Freeze all21 choices and a matched-slope control before outer queries."""
import json
from pathlib import Path
import joblib
import numpy as np
from source_oof import sa, HERE, FROZEN
from batch_random_function import checked


def build_manifest(results):
    batch = results / 'm4_random_function_batch_v1'
    b = json.loads((batch / 'result.json').read_text()); req = json.loads((batch / 'request.json').read_text())
    assert b['status'] == 'ALL_VALIDATION_REPLAY_AUDITED' and len(b['folds']) == 21
    assert {r['fold'] for r in b['folds']} == set(req['folds']) and len(set(req['folds'])) == 21
    assert all(sa.digest(HERE / n) == h for n, h in req['code_hashes'].items())
    protocol = json.loads((HERE / 'random_function_protocol.json').read_text()); assert protocol == req['protocol']
    old = results / 'm4_support_cv_screen_v1'
    ca = json.loads((old / 'verification.json').read_text()); cf = json.loads((old / 'frozen_selections.json').read_text())
    assert ca['status'] == 'PASS' and ca['summary_sha256'] == sa.digest(old / 'summary.json')
    old_choices = {r['fold']:r for r in cf['selections']}
    selections = []; source_models = []
    for entry in sorted(b['folds'], key=lambda r:r['fold']):
        fold = entry['fold']; root = Path(entry['root'])
        assert entry == checked(root, fold, protocol)
        r = json.loads((root / 'result.json').read_text()); source_req = json.loads((root / 'request.json').read_text())
        choices = {}
        for family in protocol['families']:
            selected = r['selected'][family]
            assert selected == min([t for t in r['trials'] if t['target'] == family], key=lambda t:(t['validation_mae'], t['gain'], t['key']))
            kernel = selected['key'].split('__')[1]
            if family == 'slope_uniform':
                model = next(m for m in source_req['control']['models'] if m['key'] == 'mixed__' + kernel)
                original = old_choices[fold]['choices']['uniform']
                assert model == original['model']
                assert selected['gain'] == original['selected']['gain']
                assert kernel == original['selected']['key'].split('__')[1]
            else:
                model = next(m for m in r['models'] if m['kernel'] == kernel)
            choices[family] = dict(selected=selected, model=model)
        selected = choices['function_uniform']['selected']; kernel = selected['key'].split('__')[1]
        matched = next(m for m in source_req['control']['models'] if m['key'] == 'mixed__' + kernel)
        choices['matched_slope'] = dict(selected=dict(target='matched_slope', key='matched_slope__' + kernel,
            gain=selected['gain'], selection_rule='Exact kernel/gain of primary function_uniform; no independent optimization'), model=matched)
        for c in choices.values(): assert sa.digest(Path(c['model']['path'])) == c['model']['sha256']
        for model in r['models']:
            gp = joblib.load(model['path']).gp
            theta, bounds = gp.kernel_.theta, gp.kernel_.bounds
            hit = np.flatnonzero(np.any(np.isclose(theta[:, None], bounds, atol=1e-4, rtol=0), axis=1)).tolist()
            source_models.append(dict(fold=fold, kernel=model['kernel'], optimization=model['optimization'],
                diagnostics=model['diagnostics'], kernel_bound_coordinates=hit, model_sha256=model['sha256']))
        selections.append(dict(fold=fold, choices=choices, validation_root=str(root),
            validation_result_sha256=sa.digest(root / 'result.json'), validation_audit_sha256=sa.digest(root / 'verification.json')))
    assert len(source_models) == 42
    parent = results / 'm3_waveform_candidate_v1'
    assert cf['parent_manifest_sha256'] == sa.digest(parent / 'candidate.json')
    assert cf['m2_manifest_sha256'] == sa.digest(FROZEN / 'candidate.json')
    for folder in (parent, FROZEN):
        manifest = json.loads((folder / 'candidate.json').read_text())
        assert all(sa.digest(folder / m['artifact']) == m['sha256'] for m in manifest['manifest'])
    return dict(status='ALL_CHOICES_FROZEN_NO_NEW_QUERY_SCORES', primary='function_uniform', selections=selections,
        protocol=protocol, source_models=source_models, batch_result_sha256=sa.digest(batch / 'result.json'),
        batch_request_sha256=sa.digest(batch / 'request.json'), parent_manifest_sha256=sa.digest(parent / 'candidate.json'),
        m2_manifest_sha256=sa.digest(FROZEN / 'candidate.json'), old_control_frozen_sha256=sa.digest(old / 'frozen_selections.json'),
        old_control_summary_sha256=sa.digest(old / 'summary.json'), old_control_audit_sha256=sa.digest(old / 'verification.json'),
        code_sha256=sa.digest(Path(__file__)),
        limits='Five families including independently selected and primary-matched slope controls. Original validation/development selection,not independent confirmation. All source optimizer and bound flags retained. No new outer query reads in freeze.')


def main():
    results = HERE.parent / 'results'; manifest = build_manifest(results)
    out = results / 'm4_random_function_screen_v1'; out.mkdir(exist_ok=False)
    sa.write_json(out / 'frozen_selections.json', manifest)
    print('21folds x5families frozen; no outer scores', flush=True)


if __name__ == '__main__':
    main()
