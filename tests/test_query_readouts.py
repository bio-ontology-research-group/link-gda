import unittest
import numpy as np
import torch
from phenoquery.matched_support import aggregate,cal
class Tests(unittest.TestCase):
 def test_multiplicity(self):
  x=np.array([[1.,2.],[4.,8.]],dtype='float32');p={'d':['a','a','b']};v=aggregate(x,['a','b'],p,['d'],'pool',0.)[0];np.testing.assert_array_equal(v,np.array([2.,4.],dtype='float32'))
 def test_legacy_pool_exact(self):
  x=np.random.default_rng(1).normal(size=(7,13)).astype('float32');p={'d':['a','a','c','g']};ps=list('abcdefg');block=x[[0,0,2,6]]
  for w in map(float,np.linspace(0,1,11)):np.testing.assert_array_equal(aggregate(x,ps,p,['d'],'pool',float(w))[0],w*block.max(0)+(1-w)*block.mean(0))
 def test_conjunction_rank(self):
  x=np.random.default_rng(4).normal(size=(4,7));v=aggregate(x,list('abcd'),{'d':list('abcd')},['d'],'and',1.)[0];np.testing.assert_equal(np.argsort(v),np.argsort(torch.tensor(x).sigmoid().prod(0).numpy()))
 def test_extremes(self):self.assertTrue(np.isfinite(aggregate(np.array([[1000.,-1000.]]),['a'],{'d':['a']},['d'],'and',.0625)).all())
 def test_calibration(self):
  x=np.random.default_rng(1).normal(size=(9,7));expected=[]
  for i,v in enumerate(x):
   y=np.delete(x,i,0);expected.append((v-y.mean(0))/(y.std(0)+1e-12))
  np.testing.assert_allclose(cal(x),expected,rtol=1e-10)
 def test_replay_temperatures(self):
  for inp in [.0625,1,64]:
   for tau in [.01,.1,1,10]:
    torch.manual_seed(1);w=torch.randn(3,5,dtype=torch.float64,requires_grad=True);data=torch.randn(6,3,dtype=torch.float64);x=data@w;y=torch.nn.functional.logsigmoid(x/inp).mean(0)/tau;(torch.logsumexp(y,0)-y[:2].mean()).backward();expected=w.grad.clone();w.grad=None;small=x.detach().requires_grad_(True);y=torch.nn.functional.logsigmoid(small/inp).mean(0)/tau;(torch.logsumexp(y,0)-y[:2].mean()).backward()
    for i in range(len(data)):(data[i]@w).backward(small.grad[i])
    torch.testing.assert_close(w.grad,expected)
if __name__=='__main__':unittest.main()
