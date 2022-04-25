"""
Author: Zhongqiang (Richard) Ren
All Rights Reserved.
ABOUT: Utility.
Oeffentlich fuer: RSS22
"""

import numpy as np
import matplotlib.pyplot as plt
import heapq as hpq
import matplotlib.cm as cm
import json

class PrioritySet(object):
  """
  priority queue, min-heap
  """
  def __init__(self):
    """
    no duplication allowed
    """
    self.heap_ = []
    self.set_ = set()
  def add(self, pri, d):
    """
    will check for duplication and avoid.
    """
    if not d in self.set_:
        hpq.heappush(self.heap_, (pri, d))
        self.set_.add(d)
  def pop(self):
    """
    impl detail: return the first(min) item that is in self.set_
    """
    pri, d = hpq.heappop(self.heap_)
    while d not in self.set_:
      pri, d = hpq.heappop(self.heap_)
    self.set_.remove(d)
    return pri, d
  def size(self):
    return len(self.set_)
  def print(self):
    print(self.heap_)
    print(self.set_)
    return
  def remove(self, d):
    """
    implementation: only remove from self.set_, not remove from self.heap_ list.
    """
    if not d in self.set_:
      return False
    self.set_.remove(d)
    return True


def gridAstar(grids, start, goal, w=1.0):
  """
  Four-connected Grid
  Return a path (in REVERSE order!)
  a path is a list of node ID (not x,y!)
  """
  output = list()
  (nyt, nxt) = grids.shape # nyt = ny total, nxt = nx total
  action_set_x = [-1,0,1,0]
  action_set_y = [0,-1,0,1]
  open_list = []
  hpq.heappush( open_list, (0, start) )
  close_set = dict()
  parent_dict = dict()
  parent_dict[start] = -1
  g_dict = dict()
  g_dict[start] = 0
  gx = goal % nxt
  gy = int(np.floor(goal/nxt))
  search_success = True
  while True:
    if len(open_list) == 0:
      search_success = False;
      break;
    cnode = hpq.heappop(open_list)
    cid = cnode[1]
    curr_cost = g_dict[cid]
    if cid in close_set:
      continue
    close_set[cid] = 1
    if cid == goal:
      break
    # get neighbors
    # action_idx_seq = np.random.permutation(5)
    cx = cid % nxt
    cy = int(np.floor(cid / nxt))
    for action_idx in range(len(action_set_x)):
      nx = cx + action_set_x[action_idx]
      ny = cy + action_set_y[action_idx]
      if ny < 0 or ny >= nyt or nx < 0 or nx >= nxt:
        continue
      if grids[ny,nx] > 0.5:
        continue
      nid = ny*nxt+nx
      heu = np.abs(gx-nx) + np.abs(gy-ny) # manhattan heu
      gnew = curr_cost+1
      if (nid) not in close_set:
        if (nid) not in g_dict:
          hpq.heappush(open_list, (gnew+w*heu, nid))
          g_dict[nid] = gnew
          parent_dict[nid] = cid
        else: 
          if (gnew < g_dict[nid]):
            hpq.heappush(open_list, (gnew+w*heu, nid))
            g_dict[nid] = gnew
            parent_dict[nid] = cid
  # end of while

  # reconstruct path
  if search_success:
    cid = goal
    output.append(cid)
    while parent_dict[cid] != -1 :
      cid = parent_dict[cid]
      output.append(cid)
  else:
    # do nothing
    print(" fail to plan !")
  return output

def getTargetGraph(grids,Vo,Vt,Vd):
  """
  Return a cost matrix of size |Vo|+|Vt|+|Vd| (|Vd|=|Vo|)
    to represent a fully connected graph.
  The returned spMat use node index (in list Vo+Vt+Vd) instead of node ID.
  """
  N = len(Vo)
  M = len(Vt)
  nn = N+M+N
  V = Vo + Vt + Vd
  spMat = np.zeros((nn,nn))
  for i in range(nn):
    for j in range(i+1,nn):
      spMat[i,j] = len( gridAstar(grids,V[i],V[j]) ) - 1
      spMat[j,i] = spMat[i,j]
  return spMat

def ItvOverlap(ita,itb,jta,jtb):
  """
  check if two time interval are overlapped or not.
  """
  if ita >= jtb or jta >= itb: # non-overlap
    return False, -1.0, -1.0
  # must overlap now
  tlb = jta # find the larger value among ita and jta, serve as lower bound
  if ita >= jta:
    tlb = ita
  tub = jtb # find the smaller value among itb and jtb, serve as upper bound
  if itb <= jtb:
    tub = itb
  return True, tlb, tub



