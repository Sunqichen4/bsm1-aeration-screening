"""Run release checks without overwriting scientific evidence.

From the release root: python -B run_checks.py --output runs/check_01
Add --extended for process/Gate replay and one new native BSM1 case.
This does not repeat all historical studies or optimizer runs.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory outside the scientific packages')
    parser.add_argument('--extended', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = args.output.resolve()
    protected = ['m1_m2_reconstructed_20261103', 'model_revision_20261016',
                 'boundary_validation_20261016', 'supplement_data',
                 'coefficient_evidence', 'verification',
                 'runtime_guard_reconstructed_20261104']
    if any(out.is_relative_to(root / name) for name in protected):
        parser.error('Output must be outside the supplied scientific directories')
    if out == root or out.is_relative_to(root / '.git'):
        parser.error('Choose a new runs directory or a directory outside this repository')
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['NUMBA_CACHE_DIR'] = str(out / 'runtime_cache')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        env[key] = '1'
    core = root / 'm1_m2_reconstructed_20261103'
    py = [sys.executable, '-X', 'utf8', '-B']
    commands = [
        ('release_integrity', [str(root / 'verify_release.py')], root),
        ('m1_m2_integrity', [str(core / 'reproduce.py'), 'verify', '--output', str(out / 'm1_m2_integrity')], root),
        ('m1_m2_tables', [str(core / 'reproduce.py'), 'reproduce', '--output', str(out / 'm1_m2_tables')], root),
        ('m1_m2_tests', ['-m', 'unittest', 'discover', '-s', 'tests'], core),
        ('m3_m3b_replay', [str(root / 'verification' / 'verify_m3_m3b.py'), '--output', str(out / 'm3_m3b_replay')], root),
    ]
    guard = root / 'runtime_guard_reconstructed_20261104'
    if guard.is_dir():
        commands.append(('input_guard_tests', ['-m', 'unittest', '-v', 'test_input_guard'], guard))
    if args.extended:
        for name, version, model in [('m1_prediction', 'M1', 'full'),
                                     ('m2_full_prediction', 'M2', 'full'),
                                     ('m2_ablated_prediction', 'M2', 'no_oxygen_demand_feedback')]:
            commands.append((name, [str(core / 'reproduce.py'), 'predict', '--version', version,
                                   '--model', model, '--all-cases', '--output', str(out / name)], root))
        commands.extend([
            ('m1_gate_replay', [str(core / 'reproduce.py'), 'gate-audit', '--output', str(out / 'm1_gate_replay')], root),
            ('native_reference_example', [str(core / 'reproduce.py'), 'reference-smoke', '--output', str(out / 'native_reference_example')], root),
        ])
    checks = []
    for name, arguments, cwd in commands:
        started = time.perf_counter()
        process = subprocess.run(py + arguments, cwd=cwd, env=env,
                                 capture_output=True, text=True, encoding='utf-8', errors='replace')
        (out / (name + '.stdout.log')).write_text(process.stdout, encoding='utf-8')
        (out / (name + '.stderr.log')).write_text(process.stderr, encoding='utf-8')
        item = {'check': name, 'passed': process.returncode == 0,
                'exit_code': process.returncode,
                'elapsed_seconds': round(time.perf_counter() - started, 4)}
        checks.append(item)
        print(json.dumps(item), flush=True)
        if process.returncode:
            break
    summary = {'passed': len(checks) == len(commands) and all(x['passed'] for x in checks),
               'extended': args.extended, 'python': sys.version,
               'checks': checks, 'checks_planned': len(commands),
               'scope': 'Release integrity; current replay and tests. Extended mode adds surrogate integrations, Gate replay, and one fresh constant-condition native reference. Historical dynamic, sensitivity, timing and all native batches are not rerun.',
               'fitting_executed': False,
               'fresh_native_reference_executed': args.extended and any(x['check'] == 'native_reference_example' and x['passed'] for x in checks)}
    (out / 'check_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
