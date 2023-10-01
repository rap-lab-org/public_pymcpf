"""
Author: Zhongqiang (Richard) Ren
All Rights Reserved.
ABOUT: Entrypoint to the code.
Oeffentlich fuer: RSS22
"""

import context
import time
import numpy as np
import random
import cbss_msmp
import cbss_mcpf

import common as cm

def run_CBSS_MSMP():
  """
  fully anonymous case, no assignment constraints.
  """
  print("------run_CBSS_MSMP------")
  ny = 10
  nx = 10
  grids = np.zeros((ny,nx))
  # Image coordinate is used. Think about the matrix as a 2d image with the origin at the upper left corner.
  # Row index is y and column index is x.
  # For example, in the matrix, grids[3,4] means the vertex with coordinate y=3,x=4 (row index=3, col index=4).
  grids[5,3:7] = 1 # obstacles

  # The following are vertex IDs.
  # For a vertex v with (x,y) coordinate in a grid of size (Lx,Ly), the ID of v is y*Lx+x.
  starts = [11,22,33,88,99]
  targets = [40,38,27,66,72,81,83]
  dests = [19,28,37,46,69]

  configs = dict()
  configs["problem_str"] = "msmp"
  configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.9/LKH"
  configs["time_limit"] = 60
  configs["eps"] = 0.0
  res_dict = cbss_msmp.RunCbssMSMP(grids, starts, targets, dests, configs)
  
  print(res_dict)

  return 

def run_CBSS_MCPF():
  """
  With assignment constraints.
  """
  print("------run_CBSS_MCPF------")
  ny = 10
  nx = 10
  grids = np.zeros((ny,nx))
  grids[5,3:7] = 1 # obstacles

  starts = [11,22,33,88,99]
  targets = [72,81,83,40,38,27,66]
  dests = [46,69,19,28,37]

  ac_dict = dict()
  ri = 0
  for k in targets:
    ac_dict[k] = set([ri,ri+1])
    ri += 1
    if ri >= len(starts)-1:
      break
  ri = 0
  for k in dests:
    ac_dict[k] = set([ri])
    ri += 1
  print("Assignment constraints : ", ac_dict)

  configs = dict()
  configs["problem_str"] = "msmp"
  configs["tsp_exe"] = "./pytspbridge/tsp_solver/LKH-2.0.9/LKH"
  configs["time_limit"] = 60
  configs["eps"] = 0.0

  res_dict = cbss_mcpf.RunCbssMCPF(grids, starts, targets, dests, ac_dict, configs)
  
  print(res_dict)

  return 


if __name__ == '__main__':
  print("begin of main")

  run_CBSS_MSMP()

  run_CBSS_MCPF()

  print("end of main")
