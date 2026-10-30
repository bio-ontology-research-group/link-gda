import unittest,xml.etree.ElementTree as ET
import numpy as np
import torch
from phenoquery.projection import expression,flat_entails,syntax
from phenoquery.common import aggregate,cal,ranks,eligible
from phenoquery.model import readout
class Tests(unittest.TestCase):
 def test_scope(self):
  e=('some','r',('and',(('atom','Q'),('some','s',('atom','E')))));edges=set();root=syntax(e,edges)
  self.assertIn((root,'urn:eq-syntax:exists:r',syntax(e[2],set())),edges)
  self.assertEqual(list(flat_entails('C',e)),[])
 def test_subclass_intersection_direction(self):
  self.assertEqual(set(flat_entails('C',('and',(('atom','A'),('atom','B'))))),{('C','http://subclassof','A'),('C','http://subclassof','B')})
 def test_reject_negation(self):
  xml='<owl:Class xmlns:owl="http://www.w3.org/2002/07/owl#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><owl:complementOf rdf:resource="X"/></owl:Class>'
  with self.assertRaises(ValueError):expression(ET.fromstring(xml))
 def test_and_ranking(self):
  x=torch.tensor([[1.,2.,3.],[-1.,0.,4.]],dtype=torch.float64);p=torch.tensor(0.)
  np.testing.assert_array_equal(torch.argsort(readout(x,'and',p)).numpy(),torch.argsort(x.sigmoid().prod(0)).numpy())
 def test_streaming_gradient(self):
  torch.manual_seed(2);w=torch.randn(3,4,dtype=torch.float64,requires_grad=True);data=torch.randn(5,3,dtype=torch.float64);pool=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
  for mode in ['pool','and']:
   w.grad=None;pool.grad=None;x=data@w;v=readout(x,mode,pool);loss=torch.logsumexp(v,0)-v[:2].mean();loss.backward();expected=w.grad.clone();ep=None if pool.grad is None else pool.grad.clone();w.grad=None;pool.grad=None
   small=(data@w).detach().requires_grad_(True);v=readout(small,mode,pool);(torch.logsumexp(v,0)-v[:2].mean()).backward()
   for i in range(len(data)):(data[i]@w).backward(small.grad[i])
   torch.testing.assert_close(w.grad,expected)
   if ep is not None:torch.testing.assert_close(pool.grad,ep)
 def test_numpy_torch_aggregation(self):
  x=np.random.default_rng(2).normal(size=(3,4)).astype('float32');profiles={'d':['a','b','a','c']}
  for m in ['pool','and']:
   y=aggregate(x,['a','b','c'],profiles,['d'],m,.5)[0];t=readout(torch.tensor(x,dtype=torch.float64),m,torch.tensor(0.)).numpy();np.testing.assert_allclose(y,t)
 def test_head_guard(self):
  self.assertFalse(eligible({'hits@1':.03,'hits@10':.3},{'hits@1':.04,'hits@10':.2}))
 def test_ties(self):np.testing.assert_equal(ranks(np.array([[1,1,0]]),['d'],['a','b','c'],[('a','d')]),[1.5])
if __name__=='__main__':unittest.main()
