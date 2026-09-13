"""Exact private0.5 settings for four post-screen covariance/mode controls."""
import json
from pathlib import Path
from source_oof import sa, HERE
from freeze_random_function import build_manifest as original_manifest


def build_manifest(results):
    source = results / 'm4_random_function_screen_v1'; sensitivity = results / 'm4_random_function_sensitivity_v1'
    original = json.loads((source / 'frozen_selections.json').read_text())
    assert original == original_manifest(results)
    audit = json.loads((source / 'verification.json').read_text())
    assert audit['status'] == 'PASS' and audit['summary_sha256'] == sa.digest(source / 'summary.json')
    check = json.loads((sensitivity / 'verification.json').read_text())
    assert check['status'] == 'PASS' and check['summary_sha256'] == sa.digest(sensitivity / 'summary.json')
    protocol = json.loads((HERE / 'random_function_private_controls_protocol.json').read_text())
    selections = []
    for fold in original['selections']:
        root = Path(fold['validation_root']); request = json.loads((root / 'request.json').read_text())
        choice = fold['choices']['function_private']; key = choice['selected']['key'].split('__')[1]
        slope = next(m for m in request['control']['models'] if m['key'] == 'mixed__' + key)
        assert sa.digest(Path(slope['path'])) == slope['sha256']
        choices = {}
        for family in protocol['families']:
            model = choice['model'] if family.startswith('function_') else slope
            choices[family] = dict(selected=dict(target=family, key=family + '__' + key, gain=.5 * choice['selected']['gain'],
                selection_rule='No new selection; exact private0.5 settings'), model=model)
        selections.append(dict(fold=fold['fold'], choices=choices, validation_root=str(root),
            validation_result_sha256=fold['validation_result_sha256'], validation_audit_sha256=fold['validation_audit_sha256']))
    return dict(status='ALL_CHOICES_FROZEN_NO_NEW_QUERY_SCORES', primary='function_private', protocol=protocol,
        selections=selections, source_models=original['source_models'], parent_manifest_sha256=original['parent_manifest_sha256'],
        m2_manifest_sha256=original['m2_manifest_sha256'], original_frozen_sha256=sa.digest(source / 'frozen_selections.json'),
        original_audit_sha256=sa.digest(source / 'verification.json'), sensitivity_audit_sha256=sa.digest(sensitivity / 'verification.json'),
        code_sha256=sa.digest(Path(__file__)), protocol_sha256=sa.digest(HERE / 'random_function_private_controls_protocol.json'),
        limits='Post-screen exact matching, no new fitting or parameter selection. Full-source audits retained, no independent confirmation.')


def main():
    results = HERE.parent / 'results'; manifest = build_manifest(results)
    out = results / 'm4_random_function_private_controls_v1'; out.mkdir(exist_ok=False)
    sa.write_json(out / 'frozen_selections.json', manifest)
    print('21folds x4 exact private0.5 controls frozen', flush=True)


if __name__ == '__main__': main()
