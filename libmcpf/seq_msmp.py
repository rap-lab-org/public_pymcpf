"""
Author: Zhongqiang (Richard) Ren
All Rights Reserved.
Oeffentlich fuer: RSS22 
"""

import numpy as np
import time
import copy
import os, sys
sys.path.append(os.path.abspath('../')) # to import context

import context
import tsp_wrapper
import tf_mtsp
import common as cm

class SeqMSMP(object):
  """
  The target sequencing procedure for MSMP.

    APIs required by the kbestTSP.py are
    - InitMat()
    - GenFile()
    - CallLKH()
    - ResultFromFile()
    - ChangeCost()

  """
  def __init__(self, grid, Vo, Vt, Vd, configs):
    super(SeqMSMP, self).__init__()
    self.grid = grid
    self.Vo = Vo
    self.Vt = Vt
    self.Vd = Vd
    self.V = self.Vo + self.Vt + self.Vd
    self.N = len(self.Vo)
    self.M = len(self.Vt)
    self.n2i = dict() # node ID to node index
    for i in range(len(self.V)):
      self.n2i[self.V[i]] = i
    self.infM = 999999
    self.configs = configs
    self.tsp_exe = configs["tsp_exe"]
    self.setIe = set()
    self.setOe = set()
    return

  def InitMat(self):
    self.spMat = cm.getTargetGraph(self.grid,self.Vo,self.Vt,self.Vd) # target graph, fully connected.
    self.original_spMat = copy.deepcopy(self.spMat)
    self.bigM = np.max(self.spMat)*(self.N + self.M) # totally N+M edges in a mTSP solution, not 2N+M.
    # print("bigM set to ", self.bigM)
    return

  def ChangeCost(self, v1, v2, d, ri):
    """
    input ri is not in use in MSMP.
    A generic interface for changing (directed) edge cost.
    """
    i = self.n2i[v1]
    j = self.n2i[v2]
    if d > self.bigM:
      self.spMat[i,j] = self.bigM # directed
    else:
      self.spMat[i,j] = d # directed
    return 1

  def AddIe(self, v1, v2, ri):
    """
    input ri is not in use in MSMP.
    """
    self.setIe.add(tuple([v1,v2,ri]))
    i = self.n2i[v1]
    j = self.n2i[v2]
    self.spMat[i,j] = -self.bigM # directed
    return 1

  def AddOe(self, v1, v2, ri):
    """
    input ri is not in use in MSMP.
    """
    self.setOe.add(tuple([v1,v2,ri]))
    i = self.n2i[v1]
    j = self.n2i[v2]
    self.spMat[i,j] = self.infM # directed
    return 1

  def Solve(self):
    """
    solve the instance and return the results.
    """
    tf_mat = tf_mtsp.tf_MDMTHPP(self.spMat, self.N, self.M, self.bigM, self.infM)

    if_atsp = True
    problem_str = "runtime_files/msmp"
    if problem_str in self.configs:
      problem_str = self.configs["problem_str"]
    tsp_wrapper.gen_tsp_file(problem_str, tf_mat, if_atsp)
    res = tsp_wrapper.invoke_lkh(self.tsp_exe, problem_str)
    mtsp_tours = tf_mtsp.tf_MDMTHPP_tours(res[0], self.N, self.M)
    seqs_dict = dict()
    for k in mtsp_tours:
      seqs_dict[k] = list()
      for node_idx in mtsp_tours[k]:
        seqs_dict[k].append(self.V[node_idx]) # from index to node ID.
    mtsp_costs, total_cost = tf_mtsp.tf_MDMTHPP_costs(mtsp_tours, self.original_spMat) # use the original spMat
    return 1, res[2], seqs_dict, mtsp_costs
