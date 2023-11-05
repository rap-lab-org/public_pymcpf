"""
Author: Zhongqiang (Richard) Ren
All Rights Reserved.
ABOUT: this file contains CBSS framework (abstract).
Oeffentlich fuer: RSS22
"""

import kbtsp as kb
import cbss_lowlevel as sipp
import common as cm
import copy
import time
import numpy as np
import sys

DEBUG_CBSS = 0

class CbsConstraint:
  """
  borrowed from my previous code.
  """
  def __init__(self, i, va,vb, ta,tb, j=-1, flag=-1):
    """
    create a CCbsConstraint, if a single point, then va=vb
    """
    self.i = i # i<0, iff not valid
    self.va = va
    self.vb = vb
    self.ta = ta
    self.tb = tb
    self.j = j
    self.flag = flag # flag = 1, vertex conflict, flag = 2 swap conflict

  def __str__(self):
    return "{i:"+str(self.i)+",va:"+str(self.va)+",vb:"+str(self.vb)+\
      ",ta:"+str(self.ta)+",tb:"+str(self.tb)+",j:"+str(self.j)+",flag:"+str(self.flag)+"}"

def FindPathCost(p):
  """
  find the cost of a path
  """
  last_idx = -2
  for idx in range(len(p[0])): # find last loc_id that reach goal
    i1 = len(p[0]) - 1 - idx # kth loc id
    i2 = i1-1 # (k-1)th loc id
    if i2 < 0:
      break
    if p[0][i2] == p[0][i1]:
      last_idx = i2
    else:
      break
  return p[1][last_idx] - p[1][0]

class CbsSol:
  """
  The solution in CBS high level node. A dict of paths for all robots.
  """
  def __init__(self):
    self.paths = dict()
    return

  def __str__(self):
    return str(self.paths)

  def AddPath(self, i, lv, lt):
    """
    lv is a list of loc id
    lt is a list of time
    """
    nlv = lv
    nlt = lt
    # add a final infinity interval
    nlv.append(lv[-1])
    nlt.append(np.inf)
    self.paths[i] = [nlv,nlt]
    return 

  def DelPath(self, i):
    self.paths.pop(i)
    return

  def GetPath(self, i):
    return self.paths[i]

  def CheckConflict(self, i,j):
    """
    return the first constraint found along path i and j.
    If no conflict, return empty list.
    """
    ix = 0
    while ix < len(self.paths[i][1])-1:
      for jx in range(len(self.paths[j][1])-1):
        jtb = self.paths[j][1][jx+1]
        jta = self.paths[j][1][jx]
        itb = self.paths[i][1][ix+1]
        ita = self.paths[i][1][ix]
        iva = self.paths[i][0][ix] 
        ivb = self.paths[i][0][ix+1]
        jva = self.paths[j][0][jx]
        jvb = self.paths[j][0][jx+1]
        overlaps, t_lb, t_ub = cm.ItvOverlap(ita,itb,jta,jtb)
        if not overlaps:
          continue
        if ivb == jvb: # vertex conflict at ivb (=jvb)
          return [CbsConstraint(i, ivb, ivb, t_lb+1, t_lb+1, j, 1), CbsConstraint(j, jvb, jvb, t_lb+1, t_lb+1, i, 1)] # t_ub might be inf?
          # use min(itb,jtb) to avoid infinity
        if (ivb == jva) and (iva == jvb): # swap location
          return [CbsConstraint(i, iva, ivb, t_lb, t_lb+1, j, 2), CbsConstraint(j, jva, jvb, t_lb, t_lb+1, i, 2)]
      ix = ix + 1
    return []

  def ComputeCost(self):
    """
    """
    sic = 0
    for k in self.paths:
      sic = sic + FindPathCost(self.paths[k])
    return sic

class CbssNode:
  """
  CBSS ! (Steiner)
  High level search tree node
  """
  def __init__(self, id0, sol=CbsSol(), cstr=CbsConstraint(-1,-1,-1,-1,-1,-1), c=0, parent=-1):
    """
    id = id of this high level CT node
    sol = an object of type CCbsSol.
    cstr = a list of CCbsConstraint, either empty or of length 2.
      newly added constraint in this node, to get all constraints, 
      need to backtrack from this node down to the root node.
    parent = id of the parent node of this node.
    """
    self.id = id0
    self.sol = sol
    self.cstr = cstr
    self.cost = c
    self.parent = -1 # root node
    self.root_id = -1
    return

  def __str__(self):
    str1 = "{id:"+str(self.id)+",c:"+str(self.cost)+",par:"+str(self.parent)
    return str1+",cstr:"+str(self.cstr)+",sol:"+str(self.sol)+"}"

  def CheckConflict(self):
    """
    check for conflicts along paths of all pairs of robots.
    record the first one conflict.
    Note that one conflict should be splited to 2 constraints.
    """
    done_set = set()
    for k1 in self.sol.paths:
      for k2 in self.sol.paths:
        if k2 in done_set or k2 == k1:
          continue
        # check for collision
        res = self.sol.CheckConflict(k1,k2)
        if len(res) > 0:
          # self.cstr = res # update member
          return res
      # end for k2
      done_set.add(k1)
    # end for k1
    return [] # no conflict

  def ComputeCost(self):
    """
    compute sic cost, update member, also return cost value
    """
    self.cost = self.sol.ComputeCost()
    return self.cost

class CbssFramework:
  """
  """
  def __init__(self, mtsp_solver, grids, starts, goals, dests, ac_dict, configs):
    """
    grids is 2d static grids.
    """
    self.tstart = time.perf_counter() # must be re-initialized in Search()
    self.grids = grids
    (self.yd, self.xd) = self.grids.shape
    self.starts = starts
    self.goals = goals
    self.dests = dests
    self.total_num_nodes = len(starts) + len(dests) + len(goals)
    self.num_robots = len(starts)
    self.eps = configs["eps"]
    self.configs = configs
    self.nodes = dict() # high level nodes
    self.open_list = cm.PrioritySet()
    self.closed_set = set()
    self.num_closed_low_level_states = 0
    self.total_low_level_time = 0
    self.num_low_level_calls = 0
    self.node_id_gen = 1
    self.root_set = set() # a set of all root IDs.
    self.root_seq_dict = dict() # map a root ID to its joint sequence
    self.mtsp = mtsp_solver
    self.kbtsp = kb.KBestMTSP(self.mtsp)
    self.next_seq = None
    self.eps_cost = np.inf
    return

  def BacktrackCstrs(self, nid, ri = -1):
    """
    given a node, trace back to the root, find all constraints relavant.
    """
    node_cs = list()
    swap_cs = list()
    cid = nid
    if ri < 0:
      ri = self.nodes[nid].cstr.i
    # if ri < 0, then find constraints related to robot ri.
    while cid != -1:
      # print("cid = ",cid)
      if self.nodes[cid].cstr.i == ri: # not a valid constraint
        # init call of mocbs will not enter this.
        cstr = self.nodes[cid].cstr
        if self.nodes[cid].cstr.flag == 1: # vertex constraint
          node_cs.append( (cstr.vb, cstr.tb) )
        elif self.nodes[cid].cstr.flag == 2: # swap constraint
          swap_cs.append( (cstr.va, cstr.vb, cstr.ta) )
          node_cs.append( (cstr.va, cstr.tb) ) # since another robot is coming to v=va at t=tb
      cid = self.nodes[cid].parent
    return node_cs, swap_cs
  
  def _IfNewRoot(self, curr_node):
    """
    """
    cval = curr_node.cost
    if cval > self.eps_cost:
      if not self.next_seq: # next_seq not computed yet, compute next seq
        tlimit = self.time_limit - (time.perf_counter() - self.tstart)
        flag = self.kbtsp.ComputeNextBest(tlimit, self.total_num_nodes)
        if not flag: # no joint sequence any more.
          self.next_seq = None
        else:
          self.next_seq = self.kbtsp.GetKthBestSol() # will be used to check if new root needs to be generated.
    else:
      return False

    ### if reach here, must be the case: cval > (1+eps)*curr_root_cost.
    if DEBUG_CBSS:
      print("### CBSS _IfNewRoot 2nd phase, input cost = ", cval, " eps_cost = ", self.eps_cost, " next_cost = ", (1+self.eps)*self.next_seq.cost)
    if (self.next_seq is None):
      self.eps_cost = np.inf # no next root!
      return False
    else:
      if (cval > self.next_seq.cost):
        return True
      else:
        return False

  def _GenCbssNode(self, nid):
    return CbssNode(nid)

  def _UpdateEpsCost(self, c):
    self.eps_cost = (1+self.eps)*c # update eps cost.
    # print(" _UpdateEpsCost input ", c, " eps_cost = ", self.eps_cost)
    return

  def _GenNewRoot(self):
    """
    called at the beginning of the search. 
    generate first High level node.
    compute individual optimal path for each robot.
    """

    ### Generate the first HL node, a root node ###
    nid = self.node_id_gen
    self.nodes[nid] = self._GenCbssNode(nid)
    self.node_id_gen = self.node_id_gen + 1
    self.root_set.add(nid)
    self.nodes[nid].root_id = nid

    ### Init sequencing related ###
    if not self.next_seq:
      if (nid == 1): # init
        tlimit = self.time_limit - (time.perf_counter() - self.tstart)
        flag = self.kbtsp.ComputeNextBest(tlimit, self.total_num_nodes)
        if not flag:
          print("[ERROR] CBSS: No feasible joint sequence or time out at init!")
          sys.exit("[ERROR]")
        self.root_seq_dict[nid] = self.kbtsp.GetKthBestSol() # assign seq data to root node.
      else:
        return False # no new root to be generated.
    else: # during search
      self.root_seq_dict[nid] = self.next_seq
      self.next_seq = None # next_seq has been used, make it empty.

    ### plan path based on goal sequence for all agents ###
    for ri in range(self.num_robots): # loop over agents, plan their paths
      lv, lt, stats = self.Lsearch(nid, ri)
      if len(lv) == 0: # fail to init, time out or sth.
        return False
      self.nodes[nid].sol.AddPath(ri,lv,lt)

    ### update cost and insert into OPEN ###
    c = self.nodes[nid].ComputeCost() # update node cost and return cost value
    self.open_list.add(c,nid)
    self._UpdateEpsCost(c)
    return True

  def Lsearch(self, nid, ri):
    """
    input a high level node, ri is optional.
    """
    if DEBUG_CBSS:
      print("Lsearch, nid:",nid)
    nd = self.nodes[nid]
    tlimit = self.time_limit - (time.perf_counter() - self.tstart)

    # plan from start to assigned goals and to dest as specified in goal sequence
    gseq = self.root_seq_dict[self.nodes[nid].root_id].sol[ri]
    ss = gseq[0]
    kth = 1
    t0 = 0
    all_lv = []
    all_lv.append(self.starts[ri])
    all_lt = []
    all_lt.append(0)
    success = True
    for kth in range(1, len(gseq)):
      # TODO, this can be optimized, 
      # no need to plan path between every pair of waypoints each time! Impl detail.
      gg = gseq[kth]
      ignore_goal_cstr = True
      if kth == len(gseq)-1: # last goal
        ignore_goal_cstr = False
      lv, lt, sipp_stats = self.LsearchP2P(nid, ri, ss, gg, t0, ignore_goal_cstr)
      if DEBUG_CBSS:
        print("---LsearchP2P--- for agent ", ri, ", ignore_goal_cstr = ", ignore_goal_cstr, ", lv = ", lv, ", lt = ", lt)
      if len(lv) == 0: # failed
        success = False
        break
      else: # good
        self.UpdateStats(sipp_stats)
        all_lv, all_lt = self.ConcatePath(all_lv, all_lt, lv, lt)
      ss = gg # update start for the next call
      t0 = lt[-1]
    # end for kth
    if not success:
      return [], [], success
    else:
      return all_lv, all_lt, success

  def LsearchP2P(self, nid, ri, ss, gg, t0, ignore_goal_cstr):
    """
    Do low level search for agent-i from vertex ss with starting time step t0
      to vertex gg subject to constraints in HL node nid.
    """
    nd = self.nodes[nid]
    if ri < 0: # to support init search.
      ri = nd.cstr.i
    tlimit = self.time_limit - (time.perf_counter() - self.tstart)
    ncs, ecs = self.BacktrackCstrs(nid, ri)
    # plan from start to assigned goals and to dest as specified in goal sequence
    ssy = int(np.floor(ss/self.xd)) # start y
    ssx = int(ss%self.xd) # start x
    ggy = int(np.floor(gg/self.xd)) # goal y
    ggx = int(gg%self.xd) # goal x
    res_path, sipp_stats = sipp.RunSipp(self.grids, ssx, ssy, \
      ggx, ggy, t0, ignore_goal_cstr, 1.0, 0.0, tlimit, ncs, ecs) # note the t0 here!
    if len(res_path)==0:
      return [],[],sipp_stats
    else:
      return res_path[0], res_path[1], sipp_stats
  
  def ConcatePath(self, all_lv, all_lt, lv, lt):
    """
    remove the first node in lv,lt and then concate with all_xx.
    """
    if (len(all_lt) > 0) and (lt[0] != all_lt[-1]):
      print("[ERROR] ConcatePath lv = ", lv, " lt = ", lt, " all_lv = ", all_lv, " all_lt = ", all_lt)
      sys.exit("[ERROR] ConcatePath, time step mismatch !")
    return all_lv + lv[1:], all_lt + lt[1:]

  def FirstConflict(self, nd):
    return nd.CheckConflict()

  def UpdateStats(self, stats):
    """
    """
    if DEBUG_CBSS:
      print("UpdateStats, ", stats)
    self.num_closed_low_level_states = self.num_closed_low_level_states + stats[0]
    self.total_low_level_time = self.total_low_level_time + stats[2]
    return

  def ReconstructPath(self, nid):
    """
    """
    path_set = dict()
    for i in range(self.num_robots):
      lx = list()
      ly = list()
      lv = self.nodes[nid].sol.GetPath(i)[0]
      for v in lv:
        y = int(np.floor(v / self.xd))
        x = int(v % self.xd)
        ly.append(y)
        lx.append(x)
      lt = self.nodes[nid].sol.GetPath(i)[1]
      path_set[i] = [lx,ly,lt]
    return path_set

  def _HandleRootGen(self, curr_node):
    """
    generate new root if needed
    """  
    # print(" popped node ID = ", curr_node.id)
    if self._IfNewRoot(curr_node):
      if DEBUG_CBSS:
        print(" ### CBSS _GenNewRoot...")
      self._GenNewRoot()
      self.open_list.add(curr_node.cost, curr_node.id) # re-insert into OPEN for future expansion.
      popped = self.open_list.pop() # pop_node = (f-value, high-level-node-id)
      curr_node = self.nodes[popped[1]]
    else:
      # print(" self._IfNewRoot returns false...")
      place_holder = 1
    # end of if/while _IfNewRoot
    # print("### CBSS, expand high-level node ID = ", curr_node.id)
    return curr_node

  def Search(self):
    """
    = high level search
    """
    print("CBSS search begin!")
    self.time_limit = self.configs["time_limit"]
    self.tstart = time.perf_counter()

    good = self._GenNewRoot()
    if not good:
      output_res = [ int(0), float(-1), int(0), int(0), \
        int(self.num_closed_low_level_states), 0, float(time.perf_counter()-self.tstart), \
        int(self.kbtsp.GetTotalCalls()), float(self.kbtsp.GetTotalTime()), int(len(self.root_set)) ]
      return dict(), output_res
    
    tnow = time.perf_counter()
    # print("After init, tnow - self.tstart = ", tnow - self.tstart, " tlimit = ", self.time_limit)
    if (tnow - self.tstart > self.time_limit):
      print(" FAIL! timeout! ")
      search_success = False
      output_res = [ int(0), float(-1), int(0), int(0), \
        int(self.num_closed_low_level_states), 0, float(time.perf_counter()-self.tstart), \
        int(self.kbtsp.GetTotalCalls()), float(self.kbtsp.GetTotalTime()), int(len(self.root_set)) ]
      return dict(), output_res

    search_success = False
    best_g_value = -1
    reached_goal_id = -1

    while True:
      tnow = time.perf_counter()
      rd = len(self.closed_set)
      # print("tnow - self.tstart = ", tnow - self.tstart, " tlimit = ", self.time_limit)
      if (tnow - self.tstart > self.time_limit):
        print(" FAIL! timeout! ")
        search_success = False
        break
      if (self.open_list.size()) == 0:
        print(" FAIL! openlist is empty! ")
        search_success = False
        break

      popped = self.open_list.pop() # pop_node = (f-value, high-level-node-id)
      curr_node = self.nodes[popped[1]]
      curr_node = self._HandleRootGen(curr_node) # generate new root if needed
      tnow = time.perf_counter()
      # print("tnow - self.tstart = ", tnow - self.tstart, " tlimit = ", self.time_limit)

      if (tnow - self.tstart > self.time_limit):
        print(" FAIL! timeout! ")
        search_success = False
        break

      self.closed_set.add(popped[1]) # only used to count numbers

      if DEBUG_CBSS:
        print("### CBSS popped node: ", curr_node)
      
      cstrs = self.FirstConflict(curr_node)

      if len(cstrs) == 0: # no conflict, terminates
        print("! CBSS succeed !")
        search_success = True
        best_g_value = curr_node.cost
        reached_goal_id = curr_node.id
        break

      max_child_cost = 0
      for cstr in cstrs:
        if DEBUG_CBSS:
          print("CBSS loop over cstr:",cstr)

        ### generate constraint and new HL node ###
        new_id = self.node_id_gen
        self.node_id_gen = self.node_id_gen + 1
        self.nodes[new_id] = copy.deepcopy(curr_node)
        self.nodes[new_id].id = new_id
        self.nodes[new_id].parent = curr_node.id
        self.nodes[new_id].cstr = cstr
        self.nodes[new_id].root_id = self.nodes[curr_node.id].root_id # copy root id.
        ri = cstr.i

        ### replan paths for the agent, subject to new constraint ###
        lv,lt,flag = self.Lsearch(new_id, ri)
        self.num_low_level_calls = self.num_low_level_calls + 1 
        if len(lv)==0:
          # this branch fails, robot ri cannot find a consistent path.
          continue
        self.nodes[new_id].sol.DelPath(ri)
        self.nodes[new_id].sol.AddPath(ri,lv,lt)
        nn_cost = self.nodes[new_id].ComputeCost()
        if DEBUG_CBSS:
          print("### CBSS add node ", self.nodes[new_id], " into OPEN,,, nn_cost = ", nn_cost)
        self.open_list.add(nn_cost, new_id)
        max_child_cost = np.max([nn_cost, max_child_cost])
      # end of for cstr
      # print(">>>>>>>>>>>>>>>>>>>> end of an iteration")
    # end of while

    output_res = [ int(len(self.closed_set)), float(best_g_value), int(0), int(self.open_list.size()), \
      int(self.num_closed_low_level_states), int(search_success), float(time.perf_counter()-self.tstart),\
      int(self.kbtsp.GetTotalCalls()), float(self.kbtsp.GetTotalTime()), int(len(self.root_set)) ]

    if search_success:
      return self.ReconstructPath(reached_goal_id), output_res
    else:
      return dict(), output_res
    
