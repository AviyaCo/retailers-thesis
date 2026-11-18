"""Reference and passive-capture contracts; no changed solver settings."""
from fractions import Fraction as F
import unittest
from common import core,torch,project,shipping_model,state_hash
from diagnose import observe_call,reference_rows


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core.configure_worker()

    def test_zero_capacity(self):
        self.assertEqual(project((F(2),F(1),F(-1)),F(0)),(F(0),)*3)

    def test_slack_capacity(self):
        self.assertEqual(project((F(2),F(1),F(-1)),F(4)),(F(2),F(1),F(0)))

    def test_binding_capacity(self):
        self.assertEqual(project((F(2),F(1),F(-1)),F(1)),(F(1),F(0),F(0)))

    def test_genuine_tiny_positive(self):
        tiny=F(1,10**400)
        self.assertEqual(project((tiny,F(0),F(-1)),F(1))[0],tiny)

    def test_promotion_is_exact(self):
        z=torch.tensor([1e-6,-6.357828770120477e-7,24.333],dtype=torch.float32)
        self.assertEqual(z.tolist(),z.double().tolist())

    def test_passive_hook_preserves_output(self):
        z=torch.tensor([[2.,1.,-1.],[1.,3.,1.]],dtype=torch.float32)
        cap=torch.tensor([[1.],[2.]],dtype=torch.float32)
        plain=core.NativeProjection();hooked=core.NativeProjection();record={}
        with torch.no_grad():
            q=plain(z,cap,core._SOURCE_SOLVER_ARGS)[0]
        output,_=observe_call(hooked,lambda:hooked(z,cap,core._SOURCE_SOLVER_ARGS),record)
        self.assertTrue(torch.equal(q,output[0]))
        self.assertEqual(record['z'],z.tolist())
        self.assertFalse(hooked._forward_hooks)
        self.assertFalse(hooked._forward_pre_hooks)

    def test_reference_never_overwrites_raw(self):
        c=dict(z=[[2.,1.,-1.]],capacity=[[1.]],q=[[1.,1e-15,0.]])
        r={};reference_rows(c,r)
        self.assertEqual(c['q'][0][1],1e-15)
        self.assertFalse(r['rows'][0]['support_matches'])

    def test_model_state_and_fees_unchanged(self):
        a=shipping_model.make_model(1.,[1.,1.,1.]);b=shipping_model.make_model(1.,[1.,1.,1.])
        frozen=state_hash(b);x=torch.ones((2,3),dtype=torch.float32)
        with torch.no_grad():
            expected=a.costs_and_shipments(x,x)
        observed,_=observe_call(b.proj_layer,lambda:b.costs_and_shipments(x,x),{})
        self.assertTrue(all(torch.equal(v,w) for v,w in zip(expected,observed)))
        self.assertEqual(state_hash(b),frozen)


if __name__=='__main__':
    unittest.main()