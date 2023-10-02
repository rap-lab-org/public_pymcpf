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

DEBUG_SEQ_MCPF = 0

class SeqMCPF():
  """
  This class does the transformation that converts the target sequencing 
    problem in MCPF to an ATSP and call ATSP solver.
  Note that, this implementation assumes starts, targets and destinations 
    are disjoint sets to each other.
  """

  def __init__(self, grid, starts, goals, dests, ac_dict, configs):
    """
    """
    ### make a copy
    self.grid = grid
    self.starts = starts # input is a list of length N, N = #agents.
    self.goals = goals # i.e. tasks, waypoints.
    self.dests = dests # N destinations.
    self.ac_dict = ac_dict # assignment constraints.
    # self.lkh_file_name = lkh_file_name
    self.configs = configs
    self.tsp_exe = configs["tsp_exe"]

    ### aux vars
    self.setStarts = set(self.starts)
    self.setGoals = set(self.goals)
    self.setDests = set(self.dests)
    (self.nyt, self.nxt) = self.grid.shape
    self.num_robot = len(starts)

    self.index2agent = list()
    self.index2nodeId = list()
    self.agentNode2index = dict() # map a tuple of (agent_id, node_id) to a node index.
    self.endingIdxGoal = -1 # after InitNodes, index [self.endingIdxGoal-1] in self.index2* is the last goal.
    self.cost_mat = [] # size is known after invoking InitNodes
    self.setIe = set()
    self.setOe = set()

  def InitTargetGraph(self):
    self.spMat = cm.getTargetGraph(self.grid,self.starts,self.goals,self.dests) # target graph, fully connected.
    self.V = self.starts + self.goals + self.dests
    self.n2i = dict() # node ID to node index in spMat.
    for i in range(len(self.V)):
      self.n2i[self.V[i]] = i
    self.bigM = (len(self.starts)+len(self.goals))*np.max(self.spMat)
    self.infM = 999999
    return

  def InitMat(self):
    """
    @2021-07 an API added for K-best TSP.
    """
    self.InitTargetGraph()
    self.InitNodes()
    self.InitEdges()
    return

  def InitNodes(self):
    """
    init nodes in the transformed graph
    """
    self.index2agent = list()
    self.index2nodeId = list()

    ### starts of agents
    idx = 0
    for ri in range(self.num_robot):
      self.index2agent.append(ri)
      self.index2nodeId.append(self.starts[ri])
      self.agentNode2index[(ri,self.starts[ri])] = idx
      idx = idx + 1

    ### goals allowed to be visited by each agent
    for vg in self.goals:
      agent_set = self._GetEligibleAgents(vg)
      for ri in agent_set:
        self.index2agent.append(ri)
        self.index2nodeId.append(vg)
        self.agentNode2index[(ri,vg)] = idx
        idx = idx + 1
    self.endingIdxGoal = idx

    ### dests allowed to be visited by each agent
    for vd in self.dests:
      agent_set = self._GetEligibleAgents(vd)
      for ri in agent_set:
        self.index2agent.append(ri)
        self.index2nodeId.append(vd)
        self.agentNode2index[(ri,vd)] = idx
        idx = idx + 1
    self.cost_mat = np.zeros((len(self.index2agent),len(self.index2agent)))
    return

  def IsStart(self, nid):
    """
    """
    if nid in self.setStarts:
      return True
    return False
  def IsGoal(self, nid):
    """
    """
    if nid in self.setGoals:
      return True
    return False
  def IsDest(self, nid):
    """
    """
    if nid in self.setDests:
      return True
    return False

  def _GetEligibleAgents(self, nid):
    if nid not in self.ac_dict:
      return range(self.num_robot)
    else:
      return self.ac_dict[nid]


  def GetDist(self, nid1, nid2):
    """
    """
    return self.spMat[self.n2i[nid1],self.n2i[nid2]]

  def _NextAgent(self, ri, nid):
    """
    Return the next agent after agent-ri w.r.t node nid 
    Note that assignment constraint need to be taken into consideration!
    """
    ## for starts
    if nid in self.setStarts:
      return ri
    ## for goals and dests
    if nid not in self.ac_dict:
      if ri + 1 >= self.num_robot:
        return 0
      else:
        return ri + 1
    else:
      for k in range(ri+1, self.num_robot):
        if k in self.ac_dict[nid]:
          return k
      for k in range(ri+1):
        if k in self.ac_dict[nid]:
          return k

  def _PrevAgent(self, ri, nid):
    """
    similar to _NextAgent(), inverse function.
    """
    ## for starts
    if nid in self.setStarts:
      return ri
    ## for goals and dests
    if nid not in self.ac_dict:
      if ri - 1 < 0:
        return self.num_robot-1
      else:
        return ri - 1
    else:
      for k in range(ri-1, -1, -1):
        if k in self.ac_dict[nid]:
          return k
      for k in range(self.num_robot, ri-1, -1):
        if k in self.ac_dict[nid]:
          return k

  def InitEdges(self):
    """
    compute edge costs between pair of nodes.
    """

    ### Compute big-M, an over-estimate of the optimal tour cost.

    ### PART-1, between starts to all others.
    for idx in range(self.num_robot):
      # directed, from nid1 to nid2
      nid1 = self.index2nodeId[idx] # must be a start
      ## from nid1 to another start
      for idy in range(self.num_robot):
        nid2 = self.index2nodeId[idy] # another start
        self.cost_mat[idx,idy] = self.infM # inf
      ## from nid1 to a goal, need to set both (idx,idy) and (idy,idx)
      for idy in range(self.num_robot, self.endingIdxGoal):
        nid2 = self.index2nodeId[idy] # a goal
        if self.index2agent[idy] != idx: # agent idx is only allowed to visit its own copy of goals/dests.
          self.cost_mat[idx,idy] = self.infM # inf
          self.cost_mat[idy,idx] = self.infM # inf
          continue
        else: # nid2 is a goal within agent idx's copy.
          self.cost_mat[idx,idy] = self.GetDist(nid1, nid2) + self.bigM
          self.cost_mat[idy,idx] = self.infM # inf., from agent's goal to agent's start.
      # from nid1 to a dest
      for idy in range(self.endingIdxGoal, len(self.index2agent)):
        nid2 = self.index2nodeId[idy] # a dest
        if self.index2agent[idy] != idx: # agent idx is only allowed to visit its own copy of goals/dests.
          self.cost_mat[idx,idy] = self.infM # infinity
          self.cost_mat[idy,idx] = 0 # zero-cost edge, from agent-idy's dest to agent-idx's start.
          continue
        else:
          self.cost_mat[idx,idy] = self.GetDist(nid1, nid2) + self.bigM
          self.cost_mat[idy,idx] = 0 

    ### PART-2, from goals to another goals/dests
    for idx in range(self.num_robot, self.endingIdxGoal): # loop over goals
      nid1 = self.index2nodeId[idx] # must be a goal
      # from nid1 to a goal
      for idy in range(self.num_robot, self.endingIdxGoal):
        nid2 = self.index2nodeId[idy] # another goal
        if (self._NextAgent(self.index2agent[idx],nid1) == self.index2agent[idy]):
          # agent-i's goal to agent-(i+1)'s goal or dest.
          if (nid1 == nid2):
            # same goal node, but for diff agents
            self.cost_mat[idx,idy] = 0
          else:
            self.cost_mat[idx,idy] = self.GetDist(nid1, nid2) + self.bigM
        else:
          # agent-i's goal is only connected to agent-(i+1)'s goal or dest.
          self.cost_mat[idx,idy] = self.infM

      # from nid1 to a dest, need to set both (idx,idy) and (idy,idx)
      for idy in range(self.endingIdxGoal,len(self.index2agent)):
        nid2 = self.index2nodeId[idy] # a destination
        if (self._NextAgent(self.index2agent[idx],nid1) != self.index2agent[idy]):
          # agent-i's goal is only connected to agent-(i+1)'s goal or dest.
          self.cost_mat[idx,idy] = self.infM
          self.cost_mat[idy,idx] = self.infM
        else:
          self.cost_mat[idx,idy] = self.GetDist(nid1, nid2) + self.bigM
          self.cost_mat[idy,idx] = self.infM # cannot move from dest to a goal

    ### STEP-3, from dests to another dests
    for idx in range(self.endingIdxGoal,len(self.index2agent)): # loop over dests
      nid1 = self.index2nodeId[idx] # a destination
      for idy in range(self.endingIdxGoal,len(self.index2agent)):
        nid2 = self.index2nodeId[idy] # another destination

        if (self._NextAgent(self.index2agent[idx],nid1) == self.index2agent[idy]):
          # agent-i's dest to the next agent's dest.
          if (nid1 == nid2):
            # same dest node, but for diff agents
            self.cost_mat[idx,idy] = 0
          else:
            self.cost_mat[idx,idy] = self.infM # self.GetDist(nid1, nid2) + self.const_tour_ub # the latter one is wrong, agent is not allowed to move from one dest to another dest.
        else:
          # agent-i's dest is only connected to the next agent's dest.
          self.cost_mat[idx,idy] = self.infM
    # print(self.cost_mat)
    ### 
    return

  def Solve(self):
    """
    solve the instance and return the results.
    """
    if ("mtsp_fea_check" in self.configs) and (self.configs["mtsp_fea_check"]==1):
      print("[ERROR] not implemented")
      sys.exit("[ERROR]")
      
    if_atsp = True
    problem_str = "runtime_files/mcpf"
    if problem_str in self.configs:
      problem_str = self.configs["problem_str"]
    tsp_wrapper.gen_tsp_file(problem_str, self.cost_mat, if_atsp)
    res = tsp_wrapper.invoke_lkh(self.tsp_exe, problem_str)
    # print("LKH res:", res)
    flag, seqs_dict, cost_dict = self.SeqsFromTour(res[0])
    if DEBUG_SEQ_MCPF > 3:
      print("[INFO] mtsp SeqsFromTour is ", flag)
    return flag, res[2], seqs_dict, cost_dict

  def SeqsFromTour(self, tour):
    """
    break a tour down into task sequences.
    """
    seqs = list()
    seqs_dict = dict()
    cost_dict = dict() # the cost of each agent's goal sequences

    this_agent = -1
    curr_cost = 0
    for ix in range(len(tour)):
      index = tour[ix]
      curr_agent = self.index2agent[index]
      curr_nid = self.index2nodeId[index]

      ### for debug
      if DEBUG_SEQ_MCPF:
        print("ix(",ix,"),index(",index,"),node(",curr_nid,"),agent(",curr_agent,")")

      if self.IsStart(curr_nid):
        ## start a new sequence for agent "this_agent".
        seq = list()
        seq.append(curr_nid)
        this_agent = curr_agent
        curr_cost = 0
        # print(" start a new seq, curr seq = ", seq)
      else:
        # print(" else ")
        if curr_agent == this_agent: # skip other agents' goals 
          last_nid = seq[-1]
          seq.append(curr_nid)
          curr_cost = curr_cost + self.GetDist(last_nid, curr_nid)
          # print(" else if, append curr seq, seq = ", seq)
          if self.IsDest(curr_nid):
            ## end a sequence for "this_agent"
            seqs_dict[this_agent] = seq
            cost_dict[this_agent] = curr_cost
            # print(" else if if, end seq = ", seq)

    if len(seqs_dict) != self.num_robot:
      # It is possible that after adding Ie and Oe, 
      # the instance becomes infeasible and thus the 
      # tour can not be splitted.
      # E.g. the Ie and Oe do not respect the one-in-a-set rules.
      return 0, dict(), dict()

    return 1, seqs_dict, cost_dict

  def ChangeCost(self, v1, v2, d, ri):
    """
    ri = robot ID.
    v1,v2 are workspace graph node ID.
    d is the enforced distance value between them.
    This modifies the cost of edges in the transformed graph G_TF!
    """

    ## get indices
    rj = self._PrevAgent(ri,v1)
    index1 = self.agentNode2index[(rj,v1)]
    index2 = self.agentNode2index[(ri,v2)]

    # an edge case @2021-07-10
    if d == np.inf:
      self.cost_mat[index1,index2] = self.infM # un-traversable!
      return True
    else:
      self.cost_mat[index1,index2] = d
      return True

    return False # unknown error

  def AddIe(self, v1, v2, ri):
    """
    ri is used. Ie is imposed on the transformed graph.

    Basically, each edge in a single-agent TSP tour corresponds to either an
    auxiliary edge or an edge in the mTSP solution.
    Also note that, when do partition based on the mTSP solution, the Ie and Oe
    are still added in the transformed graph (i.e. in that ATSP problem). As an
    example, look at the HMDMTSP implementation, the AddIe and AddOe interfaces
    considers the prevAgent and nextAgent and all the Ie and Oe are imposed on
    the transformed graph (represented by cost_mat in SeqMCPF class.)

    """
    rj = self._PrevAgent(ri,v1)
    index1 = self.agentNode2index[(rj,v1)]
    index2 = self.agentNode2index[(ri,v2)]
    # print("AddIe(",index1,index2,")")
    self.cost_mat[index1,index2] = -self.bigM # directed
    self.setIe.add(tuple([v1,v2,ri]))
    return 1

  def AddOe(self, v1, v2, ri):
    """
    input ri is used. Oe is imposed on the transformed graph.
    """
    rj = self._PrevAgent(ri,v1)
    index1 = self.agentNode2index[(rj,v1)]
    index2 = self.agentNode2index[(ri,v2)]
    # print("AddOe(",index1,index2,")")
    self.cost_mat[index1,index2] = self.infM # directed
    self.setOe.add(tuple([v1,v2,ri]))
    return 1
