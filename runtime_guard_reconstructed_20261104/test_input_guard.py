import copy
import unittest
import numpy as np
from input_guard import BASELINE, COLUMNS, UNITS, check_input

def case(load=.2, alpha=.95, kla=180.):
    row = BASELINE.copy()
    row[[1, 3]] *= 1 + .2 * load
    row[9] *= 1 + .35 * load
    data = np.column_stack(([0., 201.], np.tile(row, (2, 1))))
    meta = {"columns": COLUMNS.copy(), "units": UNITS.copy(), "tempmodel": False,
            "load_coordinate": load, "oxygen_transfer_factor": alpha, "commanded_kla_d_1": kla}
    return data, meta

class InputGuardTests(unittest.TestCase):
    def test_in_scope_is_not_acceptance(self):
        a,m=case(); r=check_input(a,m)
        self.assertEqual(r["status"],"in_scope")
        self.assertIsNone(r["final_compliant"])
        self.assertFalse(r["reference_executed"])
    def test_all_21_state_columns_detect_changes(self):
        a,m=case()
        for column in range(1,22):
            with self.subTest(column=column):
                b=a.copy(); b[1,column] += 1
                self.assertEqual(check_input(b,m)["status"],"fallback_required")
    def test_cold_dynamic_is_routed_without_label(self):
        a,m=case(); a[:,16]=10.; a[1,2]*=1.1
        r=check_input(a,m)
        self.assertEqual(r["status"],"fallback_required")
        self.assertFalse(r["reference_executed"])
        self.assertIsNone(r["final_compliant"])
    def test_temperature_state_on_routes(self):
        a,m=case(); m["tempmodel"]=True
        self.assertEqual(check_input(a,m)["status"],"fallback_required")
    def test_joint_and_feature_domain_boundaries(self):
        a,m=case(load=.5,alpha=.85,kla=40.)
        self.assertEqual(check_input(a,m)["status"],"in_scope")
        m["oxygen_transfer_factor"]=.84
        self.assertEqual(check_input(a,m)["status"],"fallback_required")
        a,m=case(kla=600.)
        self.assertEqual(check_input(a,m)["status"],"fallback_required")
    def test_no_label_leakage(self):
        a,m=case(); before=check_input(a,m)
        m["reference_compliant"]=True; m["reference_peak"]=0.
        self.assertEqual(before,check_input(a,m))
        m["reference_compliant"]=False; m["reference_peak"]=100.
        self.assertEqual(before,check_input(a,m))
    def test_invalid_numeric_and_flow_values_stop(self):
        a,m=case()
        for col,value in [(2,np.nan),(2,np.inf),(2,-1.),(15,0.),(15,-1.)]:
            with self.subTest(col=col,value=value):
                b=a.copy(); b[0,col]=value
                self.assertEqual(check_input(b,m)["status"],"invalid_input")
        b=a.tolist(); b[0][2]=True
        self.assertEqual(check_input(b,m)["status"],"invalid_input")
    def test_column_count_order_and_units_stop(self):
        a,m=case()
        self.assertEqual(check_input(a[:,:21],m)["status"],"invalid_input")
        b=copy.deepcopy(m); b["columns"][1],b["columns"][2]=b["columns"][2],b["columns"][1]
        self.assertEqual(check_input(a,b)["status"],"invalid_input")
        b=copy.deepcopy(m); b["units"][0]="h"
        self.assertEqual(check_input(a,b)["status"],"invalid_input")
    def test_time_invalid_stops(self):
        a,m=case()
        for times in ([1.,1.],[2.,1.],[-1.,1.]):
            b=a.copy(); b[:,0]=times
            self.assertEqual(check_input(b,m)["status"],"invalid_input")
    def test_physical_metadata_stops(self):
        a,m=case()
        for key,value in [("tempmodel","False"),("oxygen_transfer_factor",0.),("commanded_kla_d_1",float("nan")),("load_coordinate",True)]:
            b=copy.deepcopy(m); b[key]=value
            self.assertEqual(check_input(a,b)["status"],"invalid_input")

if __name__=="__main__": unittest.main()
