import sys,unittest,hashlib,json,tempfile,contextlib,io
from unittest.mock import patch
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import reproduce as r

class ReconstructionTests(unittest.TestCase):
    def test_gate_routing_cannot_read_reference_truth(self):
        grid=r.csv('M1_frozen_candidate_grid.csv')
        raw=r.m1_logistic(grid)>=.5
        baseline=r.route(grid,raw,'logistic').sort_values('candidate_id')
        poisoned=grid.assign(SNH_zero_violation=True,SNH_max_eval=-999.,reference_compliant=False,final_compliant=False)
        changed=r.route(poisoned,raw,'logistic').sort_values('candidate_id')
        np.testing.assert_array_equal(baseline.recheck_new,changed.recheck_new)
        np.testing.assert_array_equal(baseline.raw_new,changed.raw_new)

    def test_joint_extrapolation_routes_every_point_to_reference(self):
        grid=r.csv('M1_frozen_candidate_grid.csv');grid=grid[grid.scenario.eq('r_extrapolation')]
        # Use the actually preserved extrapolation scenario name if renamed.
        if not len(grid):
            all_grid=r.csv('M1_frozen_candidate_grid.csv');grid=all_grid[all_grid.load_coordinate.gt(1)]
        self.assertGreater(len(grid),0)
        routed=r.route(grid,r.m1_logistic(grid)>=.5,'logistic')
        self.assertTrue(routed.recheck_new.all())

    def test_process_predictions_ignore_injected_labels(self):
        grid=r.csv('M1_frozen_candidate_grid.csv').head(4)
        a=r.cm.steady_predictions(grid,r.read('M1_model_parameters.json'))
        b=r.cm.steady_predictions(grid.assign(target=True,SNH_max_eval=-1e12),r.read('M1_model_parameters.json'))
        np.testing.assert_array_equal(a,b)

    def test_strict_boolean_parser_rejects_missing_or_ambiguous(self):
        with self.assertRaises(ValueError):r.boolean(['true',None])
        with self.assertRaises(ValueError):r.boolean(['safe'])

    def test_M1_ablation_request_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'must_not_be_created'
            arguments=['reproduce.py','predict','--version','M1','--model','no_oxygen_demand_feedback','--output',str(output)]
            with patch.object(sys,'argv',arguments),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                r.main()
            self.assertEqual(error.exception.code,2)
            self.assertFalse(output.exists())

    def test_M1_prediction_fails_when_frozen_logistic_fixture_is_corrupt(self):
        real_csv=r.csv
        frozen_process=real_csv('M1_frozen_surrogate_predictions.csv')
        def fixture(name):
            frame=real_csv(name).copy(deep=True)
            if name=='M1_frozen_logistic_predictions.csv':
                frame['probability']=1.-frame.probability
                frame['raw_compliant']=~r.boolean(frame.raw_compliant)
            return frame
        with tempfile.TemporaryDirectory() as folder,patch.object(r,'csv',side_effect=fixture),patch.object(r.cm,'predict_cases',return_value=frozen_process):
            output=Path(folder)
            with self.assertRaisesRegex(AssertionError,'Prediction comparison failed'):
                r.predict(output,'M1','full')
            result=json.loads((output/'prediction_comparison.json').read_text(encoding='utf-8'))
            self.assertFalse(result['pass'])
            self.assertGreater(result['logistic_label_disagreements'],0)
            self.assertEqual(result['label_disagreements'],0)

    def test_integrity_rejects_corrupt_logistic_fixture_without_touching_inputs(self):
        original=ROOT/'inputs/M1_frozen_logistic_predictions.csv'
        original_hash=hashlib.sha256(original.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            temporary_root=Path(folder);(temporary_root/'inputs').mkdir()
            fixture=temporary_root/'inputs/M1_frozen_logistic_predictions.csv'
            fixture.write_bytes(original.read_bytes())
            manifest={'files':{'inputs/M1_frozen_logistic_predictions.csv':{'sha256':original_hash}}}
            (temporary_root/'copied_source_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            fixture.write_bytes(fixture.read_bytes()+b'corruption')
            output=temporary_root/'output';output.mkdir()
            with patch.object(r,'ROOT',temporary_root),self.assertRaisesRegex(AssertionError,'Input/source integrity failure'):
                r.verify(output)
            result=json.loads((output/'integrity.json').read_text(encoding='utf-8'))
            self.assertEqual(result['failures'],['inputs/M1_frozen_logistic_predictions.csv'])
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),original_hash)

if __name__=='__main__':unittest.main()
