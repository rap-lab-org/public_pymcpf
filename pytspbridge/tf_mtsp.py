"""
Author: Zhongqiang (Richard) Ren.
All Rights Reserved.
Note: Transformation-based method for Multiple Traveling Salesman Problem (mTSP).
      Include both MDMTSP and HMDMTSP.
"""

import numpy as np


def tf_MDMTHPP(spMat, N, M, bigM, infM):
  """
  Multi-Depot Multi-Terminal Hamiltonian Path Problem.
  All terminals are unassigned!
  In other words, there is no agent-target assignment constraints.
  Based on the tf algorithm for the MDMTSP problem.

  Input:
  N = #agents
  M = #targets
  spMat = the shortest path distance between any pair of nodes in depots U targets U destinations.
           and there should be N depots (i.e. starts), M targets, N destinations.
           The ID of depots should be 0 ~ N-1;
           The ID of targets should be N ~ N+M-1;
           The ID of destinations should be N+M ~ N+M+N-1.
  bigM = an upper bound of any possible tour cost. the lower the better.
  infM = the large integer that marks infinity. the large the better (avoid overflow).
  Ie, edges that must be included.
  Oe, edges that must be excluded.
  Ie and Oe are args that support K-best-TSP.
  
  Output:
  A cost matrix after transformation. This matrix is of size (N+M+N)x(N+M+N).
  """
  nn = 2*N+M
  cmat = np.zeros((nn, nn))
  for ix in range(nn):
    for iy in range(nn):
      # dest to others
      if ix >= N+M: # ix is a destination
        if iy < N: # iy is a start
          # dest to start, zero-cost edge
          cmat[ix,iy] = 0
          continue
        else:
          # dest to others, inf cost
          cmat[ix,iy]= infM
          continue
      elif iy < N: # iy is start
        if ix < N+M: # ix is not a destination
          cmat[ix,iy] = infM
          continue
      else: # all other cases
        cmat[ix,iy] = spMat[ix,iy]
    # end for iy
  # end for ix

  # for ie in Ie: # all edges that must be included.
  #   cmat[ie[0], ie[1]] = -bigM
  # for oe in Oe:
  #   cmat[oe[0], oe[1]] = infM
  
  return cmat

def tf_MDMTHPP_tours(s, N, M):
  """
  transform a single-agent TSP tour back to multiple agent tours.

  Input:
  s = single agent tour
  N = #agents
  M = #targets
  cost = cost of single-agent solution s.

  Output:
  a dict about each agent's tour, agent ID 0~(N-1), 
      each agent's tour = list of node indices in the cost matrix for LKH.
  """
  seq = list()
  seqs_dict = dict()
  first = True
  for ix in range(len(s)):
    n = s[ix]
    if (n < N) and (not first): 
      # break here, start a new sequence of targets
      if len(seq) > 0:
        seqs_dict[seq[0]] = seq
      seq = list()
    seq.append(n)
    first = False
    # end if 
  if len(seq) > 0:
    seqs_dict[seq[0]] = seq
  return seqs_dict

def tf_MDMTHPP_costs(seqs_dict, spMat):
  """
  recover each agent's cost based on each agent's tour (seqs_dict)
  """
  cost_dict = dict()
  total_cost = 0
  for k in seqs_dict:
    cost_dict[k] = 0
    for idx in range(len(seqs_dict[k])-1):
      cost_dict[k] += spMat[seqs_dict[k][idx], seqs_dict[k][idx+1]]
      total_cost += spMat[seqs_dict[k][idx], seqs_dict[k][idx+1]]
  return cost_dict, total_cost
