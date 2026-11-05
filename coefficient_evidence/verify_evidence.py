"""Verify the retained coefficient evidence without access to the old workspace.

This checks evidence integrity and recorded facts. It does not rerun the
scientific simulations or establish an independent physical coefficient basis.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    checks = []
    manifest = json.loads((ROOT/'evidence_manifest.json').read_text(encoding='utf-8'))
    for item in manifest:
        path = ROOT/item['evidence_path']
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        checks.append({'name':'evidence_sha256', 'path':item['evidence_path'], 'pass':actual == item['sha256']})
    audit = json.loads((ROOT/'coefficient_provenance_audit.json').read_text(encoding='utf-8'))
    for name, item in audit['historical_protocol_hash_checks'].items():
        path = ROOT/item['evidence_path']
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        checks.append({'name':'historical_protocol_hash_match', 'source':name,'pass':actual==item['expected']})
    historical = json.loads((ROOT/'source_evidence/historical_parameter_config.json').read_text(encoding='utf-8'))
    m1 = json.loads((ROOT/'source_evidence/M1_model_parameters.json').read_text(encoding='utf-8'))
    m2 = json.loads((ROOT/'source_evidence/M2_full_parameters.json').read_text(encoding='utf-8'))
    checks.append({'name':'historical_eight_fit_variables', 'pass':historical['calibration_protocol']['parameter_names']==audit['historical_calibration']['parameter_names'] and len(audit['historical_calibration']['parameter_names'])==8})
    checks.append({'name':'M1_fit_variables', 'pass':m1['all_parameters']==audit['M1_calibration']['listed_config_parameters'] and m1['optimized_steady_parameters']==audit['M1_calibration']['optimized_steady_parameters']})
    checks.append({'name':'M2_seven_fit_variables', 'pass':m2['fit_names']==audit['M2_calibration']['fit_names'] and len(m2['fit_names'])==7})
    source_lines = (ROOT/'source_evidence/historical_surrogate.py').read_text(encoding='utf-8').splitlines()
    expected = {207:'dSS = (influent.cod - SS) / tau - 18.0 * r_h',213:'dSNO = (2.0 - SNO) / tau + p.y_no * r_a - 0.25 * r_h',214:'dSO = (0.4 - SO) / tau + r_o2 - 0.40 * r_h - 2.50 * r_a'}
    for line, expression in expected.items():
        checks.append({'name':'historical_equation_location','line_1based':line,'pass':source_lines[line-1].strip()==expression})
    with (ROOT/'source_evidence/coefficient_sensitivity_summary.csv').open(encoding='utf-8-sig',newline='') as f:
        sensitivity = {r['variant']:r for r in csv.DictReader(f)}
    checks.extend([
        {'name':'sensitivity_17_variants','pass':len(sensitivity)==17 and all(int(r['n'])==244 for r in sensitivity.values())},
        {'name':'oxygen_a_x0.8_twenty_false_releases','pass':int(sensitivity['oxygen_a_demand_x0.8']['false_release'])==20},
        {'name':'oxygen_a_x1.2_forty_four_rejections','pass':int(sensitivity['oxygen_a_demand_x1.2']['conservative_rejection'])==44},
        {'name':'basis_not_fabricated','pass':all(not r['basis_record_found'] for r in audit['coefficient_records'])},
    ])
    result = {'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL','checks':checks,'new_scientific_simulations':0,'new_refits':0,'scope':'Evidence integrity and retained record consistency only; independent original coefficient basis remains unverified.'}
    (ROOT/'verification_20261103.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'checks':len(checks),'failures':[c for c in checks if not c['pass']]},ensure_ascii=False))
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':
    raise SystemExit(main())
