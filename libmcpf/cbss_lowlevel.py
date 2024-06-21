"""
Author: Zhongqiang (Richard) Ren
All Rights Reserved.
ABOUT: this file leverages my python MO-SIPP.
Oeffentlich fuer: RSS22
"""

import numpy as np
import heapq as hpq
import common as cm
import itertools as itt
import time
import copy
import sys
import common as cm
import os
import matplotlib.pyplot as plt

# this debug flag is due to some legacy, just leave it there.
DEBUG_MOSTASTAR = False

tasktime = 0

class SippState:
  """
  Search state for SIPP, changed MO-SIPP to single-objective.
  """
  def __init__(self, sid, loc, gval, t, tb, occupylist=[], carttag=0):
    """
    """
    self.id = sid
    self.loc = loc # location id
    self.g = gval # g-value
    self.t = t # arrival time step
    self.tb = tb # ending time step, mark the ending time of an interval
    self.occupylist  = occupylist #occupy location id
    self.carttag = carttag # equal to the number of the executed tasks
  
  def __str__(self):
    return "{id:"+str(self.id)+",loc:"+str(self.loc)+",g("+str(self.g)+"),itv(t,tb:"+str(self.t)+","+str(self.tb)+"),carttag("+str(self.carttag)+"),lenofoccupylist("+str(len(self.occupylist))+")}"

  def Equal(self, other):
    return cm.Equal(self.cost_vec, other.cost_vec)

class SippGridSpace:
  """
  Assume four connected grid graph.
  """
  def __init__(self, grid):
    """
    """
    self.grid = grid

    self.node_cstrs = dict() 
    # map a node to a set of forbidden timesteps.

    self.edge_cstrs = dict() 
    # map a node to a dict of (node, set of forbidden timesteps) key-value pairs, where timesteps are saved in a set
    # NOTE: this differs a bit from previous implementation, where key-value pairs are (forbidden time, set of nodes)
  
  def LookUpItv(self, nid, t):
    """
    given a node ID nid and a time step t, find the safe interval enclosing t.
    """
    lb = 0
    ub = np.inf
    if nid not in self.node_cstrs:
      return (lb, ub), True

    for tt in self.node_cstrs[nid]:
      if tt == t:
        print("[ERROR] LookUpItv, try to loop up a node-timestep (", nid, ",", t, ") that is unsafe !")
        return [], False
      # print(" tt = ", tt)
      if tt > t:
        ub = int( np.min([tt-1, ub]) )
      if tt < t:
        lb = int( np.max([tt+1, lb]) )
    return (lb, ub), True
  
  def GetSuccSafeItv(self, nid, nnid, tlb, tub):
    """
    return a list of safe intervals, tlist, locate at nnid that are reachable/successors from nid with safe interval [tlb, tub].
    """
    out = list()
    unsafe_tpoint = list()
    # print(" nid ", nid, " nnid ", nnid, " cstrs : ", self.node_cstrs)
    if nnid in self.node_cstrs:
      for tt in self.node_cstrs[nnid]:
        unsafe_tpoint.append(tt)
    if nid in self.edge_cstrs:
      if nnid in self.edge_cstrs[nid]:
        for tt in self.edge_cstrs[nid][nnid]:
          unsafe_tpoint.append(tt+1) ## note that this is tt+1, not tt.
    
    unsafe_tpoint.sort() ## sort from small to large
    # print("unsafe_tpoint at ", nnid, " is ", unsafe_tpoint, " tlb,tub = ", tlb, tub)

    t0 = tlb + 1 ## the earliest possible arrival time at node nnid.
    if t0 > tub:
      return out # empty list

    if len(unsafe_tpoint) > 0:
      for idx in range(len(unsafe_tpoint)):
        tk = unsafe_tpoint[idx]
        if tk <= t0:
          if tk == t0:
            t0 = tk + 1 ## earliest possible arrival time should be shift forwards.
            if t0 > tub: # everytime when t0 changes, need to check
              break
          continue
        ## now tk >= t0 + 1
        if tk <= tub + 1:
          out.append( (t0,tk-1) )
        else:
          ## tk >= tub + 2
          # if t0 <= tub + 1:
          out.append( (t0, tub+1) )
        t0 = tk + 1
        if t0 > tub: # everytime when t0 changes, need to check
          break
      ## check if inf interval is needed
      # if tub == np.inf:
      #   out.append( (t0, np.inf) )
      if t0 <= tub:
        out.append( (t0, np.inf) )
    else:
      out.append( (t0, np.inf) )
    return out
    
  def GetSuccs(self, u, v, tlb, tub):
    """
    move from u to v, with time in closed interval [tlb, tub].
    return all reachable safe intervals with earliest arrival time at node v.
    The returned arg is a list of SippState.
    NOTE that the ID and cost_vec in those returned states are invalid!
    """
    out = list()
    tlist = self.GetSuccSafeItv(u,v,tlb,tub)
    # print("tlist = ", tlist)
    for itv in tlist:
      itv0, success = self.LookUpItv(v, itv[0])
      if not success:
        sys.exit("[ERROR] GetSuccs, look up itv fails !")
        continue
      # sid, loc, gval, t, tb, occupylist=[], carttag=0
      out.append( SippState(-1, v, -1.0, int(itv[0]), itv[1] ) ) # Note the order, t, tb, ta
    return out

  def GetSuccs_ml(self, u, v, tlb, tub,newoccupylist,oldcarttag):
    """
    move from u to v, with time in closed interval [tlb, tub].
    return all reachable safe intervals with earliest arrival time at node v.
    The returned arg is a list of SippState.
    NOTE that the ID and cost_vec in those returned states are invalid!
    """
    out = list()
    tlist = self.GetSuccSafeItv(u,v,tlb,tub)
    # print("tlist = ", tlist)
    for itv in tlist:
      itv0, success = self.LookUpItv(v, itv[0])
      if not success:
        sys.exit("[ERROR] GetSuccs, look up itv fails !")
        continue
      # sid, loc, gval, t, tb, occupylist=[], carttag=0
      out.append( SippState(-1, v, -1.0, int(itv[0]), itv[1] ,newoccupylist,oldcarttag) ) #keep the same carttag
    return out



  def AddNodeCstr(self, nid, t):
    """
    """
    if nid not in self.node_cstrs:
      self.node_cstrs[nid] = set()
    self.node_cstrs[nid].add(t)
    # print(" SIPP Space add node cstr , ", nid, " t ", t)
    return 

  def AddEdgeCstr(self, u, v, t):
    """
    """
    if u not in self.edge_cstrs:
      self.edge_cstrs[u] = dict()
    if v not in self.edge_cstrs[u]:
      self.edge_cstrs[u][v] = set()
    self.edge_cstrs[u][v].add(t)
    return

class SIPP:
  """
  SIPP, modified from MO-SIPP (and remove the inheritance from MOSTA*). years ago...
  no wait action in the action_set_x,_y.
  """
  def __init__(self, grids, sx, sy, gx, gy, t0, ignore_goal_cstr, w=1.0, eps=0.0, action_set_x = [-1,0,1,0], action_set_y = [0,-1,0,1], gseq =[]):
    """
    """
    self.grids = grids
    (self.nyt, self.nxt) = self.grids.shape
    self.state_gen_id = 3 # 1 is start, 2 is goal
    self.t0 = t0 # starting time step.
    self.ignore_goal_cstr = ignore_goal_cstr
    # start state
    self.sx = sx
    self.sy = sy
    # goal state
    self.gx = gx
    self.gy = gy
    # search params and data structures
    self.weight = w
    self.eps = eps
    self.action_set_x = action_set_x
    self.action_set_y = action_set_y
    self.all_visited_s = dict() # map state id to state (sid, nid, cost vec)
    self.frontier_map = dict() # map nid to a set of sid
    self.open_list = cm.PrioritySet()
    self.f_value = dict() # map a state id to its f-value vector (np.array)
    self.close_set = set()
    self.backtrack_dict = dict() # track parents
    self.reached_goal_state_id = 0 # record the ID
    self.time_limit = 30 # seconds, default
    self.node_constr = dict()
      # a dict that maps a vertex id to a list of forbidden timestamps
    self.swap_constr = dict()
      # a dict that maps a vertex id to a dict with (forbidden time, set(forbidden next-vertex)) as key-value pair.
    self.sipp_space = SippGridSpace(grids)
    self.gseq = gseq#for multi astar
    return

  def GetHeuristic(self, s):
    """
    Manhattan distance
    """
    cy = int(np.floor(s.loc/self.nxt)) # curr y
    cx = int(s.loc%self.nxt) # curr x
    return abs(cy-self.gy) + abs(cx - self.gx)

  def GetHeuristic_ml(self, s):
    """
    sum of targets' Manhattan distance
    """
    cy = int(np.floor(s.loc/self.nxt)) # curr y
    cx = int(s.loc%self.nxt) # curr x
    hdistance = 0
    #tasktime = 1 # should be updated
    for i in self.gseq[1+s.carttag:]:
      iy = int(np.floor(i/self.nxt))
      ix = int(i%self.nxt)
      hdistance = hdistance + abs(cy-iy) + abs(cx - ix) + tasktime
      cy = iy
      cx = ix
    hdistance = hdistance - tasktime
    return hdistance

  def GetCost(self, loc, nloc, dt=1):
    """
    minimize arrival time. cost is time.
    """
    return dt

  def GetStateIdentifier(self, s):
    """
    override the method in parent class.
    """
    return (s.loc, s.tb) # this is a new attempt @2021-03-31

  def GetStateIdentifier_ml(self, s):
    """
    override the method in parent class.
    """
    return (s.loc, s.tb, s.carttag) #add carttag
  def GetNeighbors(self, s, tstart):
    """
    tstart is useless... can be deleted. 
    input a state s, compute its neighboring states.
    output a list of states.
    """
    s_ngh = list()
    cy = int(np.floor(s.loc/self.nxt)) # current x
    cx = int(s.loc%self.nxt) # current y

    # loop over all four possible actions
    for action_idx in range(len(self.action_set_x)):
      nx = cx+self.action_set_x[action_idx] # next x
      ny = cy+self.action_set_y[action_idx] # next y 
      nnid = ny*self.nxt+nx
      if (nx >= self.nxt) or (nx < 0) or (ny >= self.nyt) or (ny < 0): # out of border of grid
        continue
      if (self.grids[ny,nx] > 0): # static obstacle
        continue
      ### this is where SIPP takes effects ###
      snghs = self.sipp_space.GetSuccs(s.loc, nnid, s.t, s.tb)
      for sngh in snghs:
        if sngh.t > sngh.tb:
          sys.exit("[ERROR] state " + str(sngh) + " t > tb !!!")
        ## updat state id
        sngh.id = self.state_gen_id
        self.state_gen_id = self.state_gen_id + 1
        ## update state cost vector
        dt = 1 # @2021-04-09
        if sngh.t > s.t:
          dt = sngh.t - s.t
        sngh.g = s.g+self.GetCost(s.loc, nnid, dt)
        s_ngh.append(sngh)
      if DEBUG_MOSTASTAR:
        print(" get ngh from ", s, " to node ", nnid)
    return s_ngh, True


  def GetNeighbors_ml(self, s, tstart):
    """
    tstart is useless... can be deleted.
    input a state s, compute its neighboring states.
    output a list of states.
    do not engage carts
    use sipp
    """
    s_ngh = list()
    cy = int(np.floor(s.loc/self.nxt)) # current x
    cx = int(s.loc%self.nxt) # current y

    # loop over all four possible actions
    for action_idx in range(len(self.action_set_x)):
      nx = cx+self.action_set_x[action_idx] # next x
      ny = cy+self.action_set_y[action_idx] # next y
      nnid = ny*self.nxt+nx
      if (nx >= self.nxt) or (nx < 0) or (ny >= self.nyt) or (ny < 0): # out of border of grid
        continue
      if (self.grids[ny,nx] > 0): # static obstacle
        continue
      ### this is where SIPP takes effects ###
      snghs = self.sipp_space.GetSuccs_ml(s.loc, nnid, s.t, s.tb,[nnid],s.carttag)
      for sngh in snghs:
        if sngh.t > sngh.tb:
          sys.exit("[ERROR] state " + str(sngh) + " t > tb !!!")
        ## updat state id
        sngh.id = self.state_gen_id
        self.state_gen_id = self.state_gen_id + 1
        ## update state cost vector
        dt = 1 # @2021-04-09
        if sngh.t > s.t:
          dt = sngh.t - s.t
        sngh.g = s.g+self.GetCost(s.loc, nnid, dt)
        s_ngh.append(sngh)
      if DEBUG_MOSTASTAR:
        print(" get ngh from ", s, " to node ", nnid)

    if s.carttag < len(self.gseq) - 2:  # not include destination
      goalloc = self.gseq[s.carttag + 1]
      if s.loc == goalloc and  s.tb-s.t >= tasktime:
        sngh_t = SippState(-1, goalloc, -1.0, s.t + tasktime, s.tb, [goalloc],
                         s.carttag + 1)
        sngh_t.id = self.state_gen_id
        self.state_gen_id = self.state_gen_id + 1
        ## update state cost vector
        dt = 0  # @2021-04-09
        if sngh_t.t > s.t:
          dt = sngh_t.t - s.t
        sngh_t.g = s.g + self.GetCost(s.loc, goalloc, dt)
        s_ngh.append(sngh_t)


    return s_ngh, True


  def ReconstructPath(self, sid):
    """
    input state is the one that reached,
    return a list of joint vertices in right order.
    """
    jpath = [] # in reverse order
    tt = [] # in reverse order
    while sid in self.backtrack_dict:
      jpath.append(self.all_visited_s[sid].loc)
      tt.append(self.all_visited_s[sid].t)
      sid = self.backtrack_dict[sid]
    jpath.append(self.all_visited_s[sid].loc)
    tt.append(self.all_visited_s[sid].t)

    # reverse output path here.
    nodes = []
    times = []
    for idx in range(len(jpath)):
      nodes.append(jpath[len(jpath)-1-idx])
      times.append(tt[len(jpath)-1-idx])
    return nodes, times

  def ReconstructPath_ml(self, sid):
    """
    input state is the one that reached, 
    return a list of joint vertices in right order.
    """
    jpath = [] # in reverse order
    joccupylist = [] # in reverse order
    tt = [] # in reverse order
    while sid in self.backtrack_dict:
      jpath.append(self.all_visited_s[sid].loc)
      joccupylist.append(self.all_visited_s[sid].occupylist)
      tt.append(self.all_visited_s[sid].t)
      sid = self.backtrack_dict[sid] 
    jpath.append(self.all_visited_s[sid].loc)
    joccupylist.append(self.all_visited_s[sid].occupylist)
    tt.append(self.all_visited_s[sid].t)

    # reverse output path here.
    nodes = []
    occupylists = []
    times = []
    for idx in range(len(jpath)):
      nodes.append(jpath[len(jpath)-1-idx])
      occupylists.append(joccupylist[len(jpath)-1-idx])
      times.append(tt[len(jpath)-1-idx])

    # unload the cart
    mylen = len(occupylists[-1])-1
    myfinal = occupylists[-1]
    myfinalnode = nodes[-1]
    myfinaltime = times[-1]
    for i in range(mylen):
      occupylists.append(myfinal[i+1:])
      #print(lo[-1][i+1:])
      nodes.append(myfinalnode)
      times.append(myfinaltime+1)
      myfinaltime = myfinaltime + 1
    return nodes, times, occupylists

  def CheckReachGoal(self, s):
    """
    verify if s is a state that reaches goal and robot can stay there forever !!
    """
    if (s.loc != self.s_f.loc):
      return False
    ## s.loc == s_f.loc is True now
    if self.ignore_goal_cstr:
      return True # no need to consider constraints at goal.
    if s.loc not in self.node_constr:
      return True
    # print("s.t = ", s.t, " last cstr = ", self.node_constr[s.loc][-1])
    if s.t > self.node_constr[s.loc][-1]:
      # print("true")
      return True
    return False

  def CheckReachGoal_ml(self, s):
    """
    verify if s is a state that reaches goal and the head of robot can stay there forever !!
    """
    if (s.loc != self.s_f.loc):
      return False
    ## s.loc == s_f.loc is True now
    if self.ignore_goal_cstr:
      return True # no need to consider constraints at goal.


    if s.loc not in self.node_constr:
      return True
    # print("s.t = ", s.t, " last cstr = ", self.node_constr[s.loc][-1])
    if s.t > self.node_constr[s.loc][-1]:
      # print("true")
      return True
    return False


  def InitSearch(self):
    """
    override parent method
    """
    # start state
    self.s_o = SippState(1, self.sy*self.nxt+self.sx, 0.0, self.t0, np.inf)
    if (self.s_o.loc in self.node_constr) and len(self.node_constr[self.s_o.loc]) > 0:
      self.s_o.tb = min(self.node_constr[self.s_o.loc]) - 1 # leave the start node earlier than the min time in node constraints.
    # goal state
    self.s_f = SippState(2, self.gy*self.nxt+self.gx, -1.0, 0, np.inf)
    if (self.s_f.loc in self.node_constr) and len(self.node_constr[self.s_f.loc]) > 0:
      self.s_f.t = max(self.node_constr[self.s_f.loc]) + 1 # arrive at destination later than the max time in node constraints.
    if DEBUG_MOSTASTAR:
      print(" s_o = ", self.s_o, " self.node_constr = ", self.node_constr)
      print(" s_f = ", self.s_f, " self.node_constr = ", self.node_constr)
    self.all_visited_s[self.s_o.id] = self.s_o
    self.open_list.add(self.s_o.g + self.GetHeuristic(self.s_o), self.s_o.id)
    return

  def InitSearch_ml(self, occupylist):
    """
    override parent method
    """
    # start state
    self.s_o = SippState(1, self.sy*self.nxt+self.sx, 0.0, self.t0, np.inf, occupylist)
    if (self.s_o.loc in self.node_constr) and len(self.node_constr[self.s_o.loc]) > 0:
      self.s_o.tb = min(self.node_constr[self.s_o.loc]) - 1 # leave the start node earlier than the min time in node constraints.
    # goal state
    self.s_f = SippState(2, self.gy*self.nxt+self.gx, -1.0, 0, np.inf)
    if (self.s_f.loc in self.node_constr) and len(self.node_constr[self.s_f.loc]) > 0:
      self.s_f.t = max(self.node_constr[self.s_f.loc]) + 1 # arrive at destination later than the max time in node constraints.
    if DEBUG_MOSTASTAR:
      print(" s_o = ", self.s_o, " self.node_constr = ", self.node_constr)
      print(" s_f = ", self.s_f, " self.node_constr = ", self.node_constr)
    self.all_visited_s[self.s_o.id] = self.s_o
    #self.open_list.add(self.s_o.g + self.GetHeuristic(self.s_o), self.s_o.id)
    self.open_list.add(self.s_o.g + self.GetHeuristic_ml(self.s_o), self.s_o.id)
    return
  
  def Pruning(self, s):
    """
    """
    cfg_t = self.GetStateIdentifier(s)
    if cfg_t not in self.frontier_map:
      return False # this je is never visited before, should not prune
    elif self.frontier_map[cfg_t] <= s.g:
      return True
    return False # should not be pruned


  def Pruning_ml(self, s):
    """
    """
    cfg_t = self.GetStateIdentifier_ml(s)
    if cfg_t not in self.frontier_map:
      return False # this je is never visited before, should not prune
    elif self.frontier_map[cfg_t] <= s.g:
      #print(self.frontier_map[cfg_t],s.g)
      return True
    return False # should not be pruned

  def Search(self, time_limit=10):
    if DEBUG_MOSTASTAR:
      print(" SIPP Search begin ")
    self.time_limit = time_limit
    tstart = time.perf_counter()
    self.InitSearch()
    search_success = True
    rd = 0
    while(True):
      tnow = time.perf_counter()
      rd = rd + 1
      if (tnow - tstart > self.time_limit):
        print(" SIPP Fail! timeout! ")
        search_success = False
        break
      if self.open_list.size() == 0:
        # print(" SIPP Fail! Open empty! ")
        search_success = False
        break
      pop_node = self.open_list.pop() # ( sum(f), sid )
      curr_s = self.all_visited_s[pop_node[1]]
      if DEBUG_MOSTASTAR:
        print("##curr_s : ", curr_s, " g=", curr_s.g, " h=", self.GetHeuristic(curr_s))
      if DEBUG_MOSTASTAR:
        if rd % 1000 == 0:
          print(" search round = ", rd, " open_list sz = ", self.open_list.size(), \
            " time used = ", tnow - tstart )      
      if self.CheckReachGoal(curr_s): # check if reach goal(and robot can stay there!)
        self.reached_goal_state_id = curr_s.id
        search_success = True
        break
      # get neighbors
      ngh_ss, ngh_success = self.GetNeighbors(curr_s, tnow) # neighboring states
      if not ngh_success:
        search_success = False
        break
      # loop over neighbors
      for idx in range(len(ngh_ss)):
        ngh_s = ngh_ss[idx] 
        if DEBUG_MOSTASTAR:
          print (" -- loop ngh ", ngh_s)
        if (not self.Pruning(ngh_s)):
          self.frontier_map[self.GetStateIdentifier(ngh_s)] = ngh_s.g
          self.backtrack_dict[ngh_s.id] = curr_s.id
          fval = ngh_s.g + self.weight * self.GetHeuristic(ngh_s)
          self.open_list.add(fval, ngh_s.id)
          self.all_visited_s[ngh_s.id] = ngh_s
        else:
          if DEBUG_MOSTASTAR:
            print(" XX dom pruned XX ")
    if search_success:
      # output jpath is in reverse order, from goal to start
      sol_path = self.ReconstructPath(self.reached_goal_state_id)
      search_success = 1
      output_res = ( int(rd), int(search_success), float(time.perf_counter()-tstart) )
      return sol_path, output_res
    else:
      output_res = ( int(rd), int(search_success), float(time.perf_counter()-tstart) )
      return list(), output_res

  def Search_ml(self, time_limit=10, s_occupylist=[], gseq=[]):
    if DEBUG_MOSTASTAR:
      print(" multi label Search begin ")
    self.time_limit = time_limit
    tstart = time.perf_counter()
    self.InitSearch_ml(s_occupylist)
    search_success = True
    rd = 0
    while (True):
      tnow = time.perf_counter()
      rd = rd + 1
      if (tnow - tstart > self.time_limit):
        print(" multi label Search Fail! timeout! ")
        search_success = False
        break
      if self.open_list.size() == 0:
        # print(" SIPP Fail! Open empty! ")
        search_success = False
        break
      pop_node = self.open_list.pop()  # ( sum(f), sid )
      curr_s = self.all_visited_s[pop_node[1]]
      if DEBUG_MOSTASTAR:
        print("##curr_s : ", curr_s, " g=", curr_s.g, " h=", self.GetHeuristic_ml(curr_s))
        print(curr_s.occupylist)
      if DEBUG_MOSTASTAR:
        if rd % 1000 == 0:
          print(" search round = ", rd, " open_list sz = ", self.open_list.size(), \
                " time used = ", tnow - tstart)
      if self.CheckReachGoal_ml(curr_s) and curr_s.carttag == len(
              self.gseq) - 2:  # check if reach goal(and robot can stay there!)
        self.reached_goal_state_id = curr_s.id
        search_success = True
        break

      # get neighbors
      ngh_ss, ngh_success = self.GetNeighbors_ml(curr_s, tnow)  # neighboring states
      if not ngh_success:
        search_success = False
        break
      # loop over neighbors
      for idx in range(len(ngh_ss)):
        ngh_s = ngh_ss[idx]

        if DEBUG_MOSTASTAR:
          print (" -- loop ngh ", ngh_s)
        #if self.conflict_ml(curr_s, ngh_s):  # avoid the conflict
        if (not self.Pruning_ml(ngh_s)):
          self.frontier_map[self.GetStateIdentifier_ml(ngh_s)] = ngh_s.g
          self.backtrack_dict[ngh_s.id] = curr_s.id
          fval = ngh_s.g + self.weight * self.GetHeuristic_ml(ngh_s)
          self.open_list.add(fval, ngh_s.id)
          self.all_visited_s[ngh_s.id] = ngh_s
        else:

          if DEBUG_MOSTASTAR:
            print(" XX dom pruned XX ", self.GetStateIdentifier_ml(ngh_s))
    if search_success:
      # output jpath is in reverse order, from goal to start
      sol_path = self.ReconstructPath_ml(self.reached_goal_state_id)
      search_success = 1
      output_res = (int(rd), int(search_success), float(time.perf_counter() - tstart))
      return sol_path, output_res
    else:
      output_res = (int(rd), int(search_success), float(time.perf_counter() - tstart))
      return list(), output_res

  def AddNodeConstrBase(self, nid, t):
    """
  	This one is borrowed from MOSTA*, may be redundant...
  	"""
    # a new node id
    t = int(t) # make sure int!
    if nid not in self.node_constr:
      self.node_constr[nid] = list()
      self.node_constr[nid].append(t)
      # print(" AddVertexConstr - self.node_constr[",nid,"]=",self.node_constr[nid])
      return
    # locate the index for t
    idx = 0
    while idx < len(self.node_constr[nid]):
      if t <= self.node_constr[nid][idx]:
        break
      idx = idx + 1
    # if just put at the end
    if idx == len(self.node_constr[nid]):
      self.node_constr[nid].append(t)
      # print(" AddVertexConstr - self.node_constr[",nid,"]=",self.node_constr[nid])
      return
    # avoid duplication
    if t == self.node_constr[nid][idx]: 
      # print(" AddVertexConstr - self.node_constr[",nid,"]=",self.node_constr[nid])
      return
    # update list
    tlist = list()
    for idy in range(len(self.node_constr[nid])):
      if idy == idx:
        tlist.append(t)
      tlist.append(self.node_constr[nid][idy])
    self.node_constr[nid] = tlist
    # print(" AddVertexConstr - self.node_constr[",nid,"]=",self.node_constr[nid])
    return

  def AddNodeConstr(self, nid, t):
    """
    robot is forbidden from entering nid at time t.
    This is a naive implementation using list. 
    There is space for improvement on data-structure and sorting but expect to be a minor one.
    """
    if DEBUG_MOSTASTAR:
      print("*** AddNodeCstr node ", nid, " t ", t)
    self.AddNodeConstrBase(nid, t)
    self.sipp_space.AddNodeCstr(nid, t)

    return

  def AddSwapConstr(self, nid1, nid2, t):
    """
    robot is forbidden from transfering from (nid1,t) to (nid2,t+1).
    """
    # if nid1 is new to self.swap_constr[]
    if nid1 not in self.swap_constr:
      self.swap_constr[nid1] = dict()
    if t not in self.swap_constr[nid1]:
      self.swap_constr[nid1][t] = set()
    self.swap_constr[nid1][t].add(nid2) # just add 
    # print("...AddSwapConstr, self.swap_constr[",nid1,"][",t,"]=", self.swap_constr[nid1][t])
    self.sipp_space.AddEdgeCstr(nid1, nid2, t)
    return

def EnforceUnitTimePath(lv,lt):
  """
  Given a path (without the final node with infinite timestamp), 
   insert missing (v,t) to ensure every pair of adjacent (v,t) 
   has time difference of one.
  """
  dt = 1
  nlv = list()
  nlt = list()
  for ix in range(len(lt)-1):
    nlv.append(lv[ix])
    nlt.append(lt[ix])
    if lt[ix+1]-lt[ix] > 1.001:
      ct = lt[ix]
      while lt[ix+1] - ct > 1.001:
        nlv.append(lv[ix])
        nlt.append(ct+1)
        ct = ct + 1
  # end for
  nlv.append(lv[-1])
  nlt.append(lt[-1])
  return nlv, nlt

def EnforceUnitTimePath_ml(lv,lt,locp):
  """
  Given a path (without the final node with infinite timestamp),
   insert missing (v,t) to ensure every pair of adjacent (v,t)
   has time difference of one.
  """
  dt = 1
  nlv = list()
  nlt = list()
  nlocp = list()
  for ix in range(len(lt)-1):
    if lt[ix]!=lt[ix+1]:
      nlv.append(lv[ix])
      nlt.append(lt[ix])
      nlocp.append(locp[ix])
      if lt[ix+1]-lt[ix] > 1.001:
        ct = lt[ix]
        while lt[ix+1] - ct > 1.001:
          nlv.append(lv[ix])
          nlt.append(ct+1)
          nlocp.append(locp[ix])#execute the task
          ct = ct + 1
  # end for
  nlv.append(lv[-1])
  nlt.append(lt[-1])
  nlocp.append(locp[-1])
  return nlv, nlt, nlocp

def RunSipp(grids, sx, sy, gx, gy, t0, ignore_goal_cstr, w, eps, time_limit, node_cstrs=[], swap_cstrs=[]):
  """
  TODO[Done@2021-05-26], from implementation perspective, may need to consider igonring the node constraint at goal.
  Agent does not need to stay at goal after reaching it if this goal is not the destination in a MSMP problem.
  """
  if DEBUG_MOSTASTAR:
    print("...RunSipp... ")
    print("sx:", sx, " sy:", sy, " gx:", gx, " gy:", gy, " ignore_goal_cstr:",ignore_goal_cstr)
    print("node_cstrs:", node_cstrs, " swap_cstrs:", swap_cstrs)

  sipp = SIPP(grids, sx,sy,gx,gy, t0, ignore_goal_cstr, w, eps)
  for node_cstr in node_cstrs:
    sipp.AddNodeConstr(node_cstr[0], node_cstr[1])
  for swap_cstr in swap_cstrs:
    sipp.AddSwapConstr(swap_cstr[0], swap_cstr[1], swap_cstr[2])
  sol_path, stats = sipp.Search(time_limit)
  if len(sol_path) != 0:
    unit_time_path = EnforceUnitTimePath(sol_path[0], sol_path[1])
    return unit_time_path, stats
  else:
    return sol_path, stats

def RunSipp_ml(grids, gseq, t0, ignore_goal_cstr, w, eps, time_limit, node_cstrs=[], swap_cstrs=[]):
  """
  run multi label A star
  """

  (nyt, nxt) = grids.shape
  sx = gseq[0] % nxt
  sy = int(np.floor(gseq[0] / nxt))
  gx = gseq[-1] % nxt
  gy = int(np.floor(gseq[-1] / nxt))
  if DEBUG_MOSTASTAR:
    print(gseq)
    print("...Run multi a star... ")
    print("sx:", sx, " sy:", sy, " gx:", gx, " gy:", gy, " ignore_goal_cstr:",ignore_goal_cstr)
    print("node_cstrs:", node_cstrs, " swap_cstrs:", swap_cstrs)

  sipp = SIPP(grids, sx,sy,gx,gy, t0, ignore_goal_cstr, w, eps,[-1,0,1,0], [0,-1,0,1], gseq)
  for node_cstr in node_cstrs:
    sipp.AddNodeConstr(node_cstr[0], node_cstr[1])
  for swap_cstr in swap_cstrs:
    sipp.AddSwapConstr(swap_cstr[0], swap_cstr[1], swap_cstr[2])

  occupylist = [gseq[0]]
  sol_path, stats = sipp.Search_ml(time_limit, occupylist, gseq) #mulit_Search_dl_ML_sipp mulit_Search_dl
  #print(sol_path)

  if len(sol_path) != 0:
    unit_time_path = EnforceUnitTimePath_ml(sol_path[0], sol_path[1], sol_path[2])
    return unit_time_path, stats
  else:
    return sol_path, stats

# def Test1():
#   """
#   """
#   grids = np.zeros((5, 5))
#   gseq = [2,4,10]
#
#   t0 = 0
#   ignore_goal_cstr = False
#   w = 3.0
#   eps = 0.0
#   tlimit = 60
#   ncs = [(4,2)]
#   ecs = []
#   print(grids)
#
#   sol_path, stats = RunSipp_ml(grids, gseq,  t0, ignore_goal_cstr, w, eps, tlimit, ncs,ecs)
#   print(sol_path)
#   print(stats)
#   return

if __name__ == "__main__":
  Test1()


