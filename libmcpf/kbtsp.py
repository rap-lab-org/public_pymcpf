"""
Author: Zhongqiang (Richard) Ren
ABOUT: this file contains K-best partition.
Oeffentlich fuer: RSS22 
"""

import copy
import numpy as np
import sys
import time

import common as cm

DEBUG_KBESTTSP = False

class RestrictedTSP:
  """
  A restricted TSP instance.
  A tuple of (setI, setO, solution) and other auxiliary variables.
  """
  def __init__(self, setI=set(), setO=set()):
    """
    """
    self.node_id = -1
    self.setI = setI # must include
    self.setO = setO # must exclude
    self.sol = list() # solution, a dict that encodes a joint sequence
    self.cost = np.inf # tour cost.
    self.cost_dict = dict() # cost of each individual sequence
    return
  def __str__(self):
    return "<id:"+str(self.node_id)+",I:"+str(self.setI)+",O:"+str(self.setO)+\
      ",sol:"+str(self.sol)+",cost:"+str(self.cost)+",cdict:"+str(self.cost_dict)+">"

class KBestMTSP:
  """
  Compute K cheapest solutions for a mTSP.
  Note: The auxilliary edge in the transformation can be skipped when doing partition.
        Which is equivalent as doing partition directly on this mTSP solution.
  Note: Edges are considered to be directed, since after transformation the transformed
        graph is directed and corresponds to an ATSP.
  """
  def __init__(self, mtsp):
    """
    tsp = an object of mTSP solver.
    the following API of self.tsp:
      - InitMat()
      - Solve()
      - AddIe(), AddOe()
    """
    self.tsp = mtsp # mTSP solver.
    self.open_list = cm.PrioritySet()
    self.all_nodes = dict()
    self.node_id_gen = 1
    self.last_nid = -1 # this stores the k-th solution, which is not expanded yet.
    self.kbest_node = list() # the k-best nodes containing solutions.
    self.n_tsp_call = 0
    self.n_tsp_time = 0
    return

  def _Init(self):
    """
    Generate the initial solution (a node) for a TSP problem. Insert into OPEN.
    """
    
    ### generate initial restricted TSP problem instance.
    self.tsp.InitMat()
    tnow = time.perf_counter()
    flag, lb, seqs_dict, cost_dict = self.tsp.Solve()
    if flag == False:
      print("[ERROR] infeasible case? KBestTSP._Init fails to get a feasible joint sequence!")
    dt = time.perf_counter() - tnow
    self.n_tsp_call = self.n_tsp_call + 1
    self.n_tsp_time = self.n_tsp_time + dt # this is total time.

    ### generate a restricted TSP instance
    cval = np.sum(list(cost_dict.values()))
    rtsp = RestrictedTSP(set(), set())
    rtsp.sol = seqs_dict
    rtsp.cost_dict = cost_dict
    rtsp.cost = cval
    rtsp.node_id = self.node_id_gen
    self.node_id_gen = self.node_id_gen + 1
    
    ### insert into OPEN.
    self.all_nodes[rtsp.node_id] = rtsp
    self.open_list.add(cval, rtsp.node_id)
    self.last_nid = rtsp.node_id
    return rtsp

  def _SolveRTSP(self, rtsp):
    """
    modify distance matrix based on setI, setO, solve RTSP.
    """

    if DEBUG_KBESTTSP:
      print("  >> _SolveRTSP:", rtsp)

    ### copy and generate a new instance
    temp_tsp = copy.deepcopy(self.tsp)
    for ek in rtsp.setI:
      temp_tsp.AddIe(ek[0], ek[1], ek[2])
    for ek in rtsp.setO:
      temp_tsp.AddOe(ek[0], ek[1], ek[2])

    ## solve the RTSP instance
    tnow = time.perf_counter()
    success, cost_lb, seqs_dict, cost_dict = temp_tsp.Solve()
    if not success:
      # print(" temp_tsp solve success = ", success)
      return success, [], [], []

    dt = time.perf_counter() - tnow
    self.n_tsp_call = self.n_tsp_call + 1
    self.n_tsp_time = self.n_tsp_time + dt # this is total time.
    cval = np.sum(list(cost_dict.values()))

    ### verify against Ie, Oe.
    flag = self._VerifySol(temp_tsp, rtsp.setI, rtsp.setO, seqs_dict)
    if DEBUG_KBESTTSP:
      print("[INFO] kbtsp._SolveRTSP seqs_dict = ", seqs_dict, " cost_dict = ", cost_dict)
      print("[INFO] kbtsp._VerifySol returns = ", flag)
    return flag, cval, seqs_dict, cost_dict

  def _VerifySol(self, temp_tsp, setI, setO, seqs_dict):
    """
    verify whether the solution satisfies the requirement imposed by set I and O.
    """
    tempI = copy.deepcopy(setI)
    for ri in seqs_dict:
      seq = seqs_dict[ri]
      for idx in range(1,len(seq)):
        ek = tuple([seq[idx-1], seq[idx], ri])
        if ek in setO:
          if DEBUG_KBESTTSP:
            print("[INFO] kbtsp._VerifySol edge ", ek, " violates Oe ", setO)
          return False
        if ek in tempI:
          tempI.remove(ek)
    if len(tempI) > 0:
      if DEBUG_KBESTTSP:
        print("[INFO] kbtsp._VerifySol subset of Ie ", tempI, " not included")
      return False
    if hasattr(self.tsp, "verify"):
      if not temp_tsp.verify():
        return False
    return True

  def _Expand(self, nid, tlimit):
    """
    Partition over the input node (represented by its ID), 
    generate successor restricted TSP problems and insert each solution (node) into OPEN.
    """
    nk = self.all_nodes[nid]
    setI = copy.deepcopy(nk.setI)
    flag = True
    for ri in nk.sol: # nk.sol is a joint sequence. loop over each agent.
      seq = nk.sol[ri]
      for idx in range(1,len(seq)):
        ek = tuple([seq[idx-1], seq[idx], ri])
        ### Solve RTSP, setI = {e1,e2,...e(k-1)}, setO = {ek}
        setO = copy.deepcopy(nk.setO)
        setO.add(ek)
        rtsp = RestrictedTSP(copy.deepcopy(setI), setO)
        rtsp.node_id = self.node_id_gen
        self.node_id_gen = self.node_id_gen + 1
        if (time.perf_counter() - self.tstart) > tlimit:
          return False
        if not self._FeasibilityCheck1(rtsp):
          continue # the generated rtsp is obviously infeasible
        # Note: if reach here, the rtsp is not guaranteed to be feasible.
        flag, cval, seqs_dict, cost_dict = self._SolveRTSP(rtsp)
        if DEBUG_KBESTTSP:
          print("[INFO] kbtsp._Expand, RTSP get lag = ", flag, ", cost = ", cval, ", Ie = ", setI, ", Oe = ", setO)
        if flag == True: # there is such a solution.
          rtsp.sol = seqs_dict
          rtsp.cost = cval
          rtsp.cost_dict = cost_dict
          ### insert into OPEN
          self.all_nodes[rtsp.node_id] = rtsp
          self.open_list.add(cval, rtsp.node_id)
          if DEBUG_KBESTTSP:
            print("[INFO] kbtsp._Expand, add OPEN, rtsp id = ", rtsp.node_id, ", cval = ", cval)
        else:
          if DEBUG_KBESTTSP:
            print("[INFO] kbtsp._Expand, not add OPEN, rtsp id = ")
        # update setI
        setI.add(ek) # for next iteration
    return True

  def _FeasibilityCheck1(self, rtsp):
    """
    """
    for ek in rtsp.setO:
      if ek in rtsp.setI:
        print("[INFO] kbtsp._FeaCheck1, filtered ! ", ek, rtsp.node_id)
        return False
    return True

  def ComputeNextBest(self, tlimit):
    """
    compute the (self.k_count+1)-th best solution.
    """
    self.tstart = time.perf_counter()
    if len(self.kbest_node) == 0:
      self._Init()
    else:
      flag = self._Expand(self.last_nid, tlimit)
      if not flag: # timeout !
        print("[INFO] K-Best, ComputeNextBest Timeout!")
        return False
    if self.open_list.size() > 0:
      (ck, nid) = self.open_list.pop()
      # update the kth best solution and related data.
      self.the_kth_node = self.all_nodes[nid]
      self.kbest_node.append(copy.deepcopy(self.the_kth_node))
      self.last_nid = nid
      # print("[INFO] K-Best, return a next-best solution with cost = ", self.the_kth_node.cost)
      return True
    else:
      # print("[TODO] ComputeNextBest, OPEN List empty!!")
      return False

  def ComputeKBest(self, k):
    """
    compute the k-best solution from scratch.
    Invoke NextBest for k times.
    """
    # invoke NextBest()
    # TODO
    return

  def GetKBestList(self):
    """
    """
    return self.kbest_node

  def GetKthBestSol(self):
    """
    """
    return self.kbest_node[-1]

  def GetTotalTime(self):
    """
    return the total run time of TSP.
    """
    return self.n_tsp_time

  def GetTotalCalls(self):
    """
    return the total number of TSP calls.
    """
    return self.n_tsp_call
