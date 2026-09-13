"""Evidence gate for end-to-end export, not an adoption or novelty claim."""
import json
from pathlib import Path
from source_oof import sa, HERE
from freeze_random_function_private_controls import build_manifest


def evidence_gate(results):
    names = {
        'controls':'m4_random_function_private_controls_v1',
        'sensitivity':'m4_random_function_sensitivity_v1',
        'risk':'m4_random_function_risk_v1',
        'fit':'m4_random_function_fit_screen_v1'}
    summaries = {}; evidence = {}
    for key, name in names.items():
        root = results / name
        s = json.loads((root / 'summary.json').read_text()); a = json.loads((root / 'verification.json').read_text())
        assert a['status'] == 'PASS' and a['summary_sha256'] == sa.digest(root / 'summary.json')
        summaries[key] = s
        evidence[key] = dict(root=str(root), summary_sha256=sa.digest(root / 'summary.json'), audit_sha256=sa.digest(root / 'verification.json'))
    control = results / names['controls']
    frozen = json.loads((control / 'frozen_selections.json').read_text()); assert frozen == build_manifest(results)
    assert summaries['controls']['rows'] == 14600 and len(summaries['controls']['gates']['function_private']) == 32
    assert all(summaries['controls']['gates']['function_private'].values())
    assert 'function_private__0.5' in summaries['sensitivity']['passing_candidates']
    assert summaries['risk']['rows'] == 175200
    risk = summaries['risk']['pass_counts']['0.5']
    assert risk['independent'] == risk['full'] == risk['all_m4_additions'] == 10
    # All32 risk failures remain in evidence; they are not silently discarded.
    assert set(summaries['fit']['gates']) == {'init_1.0', 'init_0.5', 'init_0.8', 'init_1.25', 'init_2.0'}
    assert all(len(g) == 32 and all(g.values()) for g in summaries['fit']['gates'].values())
    fit_frozen = json.loads((results / names['fit'] / 'frozen_selections.json').read_text())
    assert len(fit_frozen['model_flags']) == 105
    assert all(m['optimization']['success'] for m in fit_frozen['model_flags'])
    uncertainty = results / 'm4_random_function_uncertainty_v1/summary.json'
    u = json.loads(uncertainty.read_text())
    assert u['status'] == 'PAIRED_DEVELOPMENT_INTERVALS_COMPLETE' and len(u['pairs']) == 1080
    assert u['summary_sha256'] == evidence['sensitivity']['summary_sha256']
    assert u['audit_sha256'] == evidence['sensitivity']['audit_sha256']
    assert u['cells_sha256'] == sa.digest(results / names['sensitivity'] / 'cells.csv')
    assert u['code_sha256'] == sa.digest(HERE / 'random_function_uncertainty.py')
    evidence['uncertainty'] = dict(path=str(uncertainty), sha256=sa.digest(uncertainty))
    root = results / 'm4_random_function_end_to_end_pilot_v1'
    pilot = json.loads((root / 'verification.json').read_text())
    assert pilot['status'] == 'ONE_CELL_END_TO_END_PASS_NOT_ADOPTED' and len(pilot['manifest']) == 8
    assert max(pilot['errors'].values()) < 1e-8 and pilot['request_sha256'] == sa.digest(root / 'request.json')
    assert all(sa.digest(root / r['artifact']) == r['sha256'] for r in pilot['manifest'])
    evidence['end_to_end_pilot'] = dict(root=str(root), audit_sha256=sa.digest(root / 'verification.json'))
    return dict(status='READY_FOR_END_TO_END_EXPORT_NOT_ADOPTED', candidate='function_private', global_step=.5,
        frozen_choices=frozen['selections'], parent_manifest_sha256=frozen['parent_manifest_sha256'],
        m2_manifest_sha256=frozen['m2_manifest_sha256'], evidence=evidence,
        point_all32_pass=True, risk_counts=risk, initialization_all32_passes=5,
        numerical_bounds=[m for m in fit_frozen['model_flags'] if m['kernel_bound_coordinates'] or m['diagnostics']['rho_bound_hit']],
        code_sha256=sa.digest(Path(__file__)),
        remaining=['Export all168 actual parent+M4 chains; all365cell end-to-end and serialization replay',
                   'Independent exported-candidate integrity/metric audit and readable report', 'Explicit experimental selection record only after export audits'],
        limits=['Exploratory post-screen private0.5 nomination; original screen primary was function_uniform.',
                'XJTU and MATR have incremental confidence intervals crossing zero; not all-dataset significance.',
                'Risk all32 passes8of10, with M2-related failures reported, although all8M4 additions pass10of10.',
                'Not superior to every matched-control metric; no independent confirmation or mathematical novelty claim.',
                'Source initialization sensitivity is not whole-pipeline random retraining.',
                'Original M1/M2/M3 unchanged, CALCE excluded from selection.'])


def main():
    result = evidence_gate(HERE.parent / 'results')
    out = HERE.parent / 'results/m4_random_function_export_gate_v1'; out.mkdir(exist_ok=False)
    sa.write_json(out / 'evidence.json', result)
    print(result['status'], flush=True)


if __name__ == '__main__': main()
