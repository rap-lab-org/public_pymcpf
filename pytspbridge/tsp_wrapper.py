
import subprocess
import numpy as np
import sys
import os

def gen_tsp_file(problem_str, cmat, if_atsp):
  """
  tsp_fn = tsp file name, the output path.
  cmat = cost matrix
  if_atsp = If the instance is ATSP (1 or True) or TSP (0 or False)
  """
  nx,ny = cmat.shape
  if nx != ny or nx == 0:
    sys.exit("[ERROR] _gen_tsp_file, input cmat is not a square matrix or of size 0!")
  with open(problem_str+".tsp", mode="w+") as ftsp:
    ### generate file headers
    if if_atsp:
      ftsp.writelines(["NAME : mtspf\n", "COMMENT : file for mtspf test\n", "TYPE : ATSP\n"])
    else:
      ftsp.writelines(["NAME : mtspf\n", "COMMENT : file for mtspf test\n", "TYPE : TSP\n"])
    ftsp.write("DIMENSION : "+str(nx)+ "\n")
    ftsp.writelines(["EDGE_WEIGHT_TYPE : EXPLICIT\n", "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n", "EDGE_WEIGHT_SECTION\n"])
    ### generate cost matrix
    for ix in range(nx):
      nline = ""
      for iy in range(nx):
        nline = nline + str( int(cmat[(ix,iy)]) ) + " " 
      ftsp.write(nline+"\n")
    ftsp.close()
  # end with
  return 0

def gen_par_file(problem_str):
  with open(problem_str+".par", mode="w+") as fpar:
    fpar.writelines(["PROBLEM_FILE = "+problem_str+".tsp\n"])
    fpar.writelines(["MOVE_TYPE = 5\n"])
    fpar.writelines(["PATCHING_C = 3\n"])
    fpar.writelines(["PATCHING_A = 2\n"])
    fpar.writelines(["RUNS = 10\n"])
    fpar.writelines(["OUTPUT_TOUR_FILE = "+problem_str+".tour\n"])
    fpar.close()
  return

def invoke_lkh(exe_path, problem_str):
  """
  call LKH executable at exe_path
  problem_str = the path the identifies the problem, without any suffix. 
    E.g. path/to/file/pr2392 (instead of pr2392.tsp)
  """

  ### generate par file for LKH if it does not exist.
  if not os.path.isfile(problem_str+".par"):
    print("[INFO] invoke_lkh, generate par file...")
    gen_par_file(problem_str)

  ### call LKH
  lb_cost = np.inf # meaningless number
  cmd = [exe_path, problem_str+".par"]
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
  for line in process.stdout:
    # print(line)
    line_str = line.decode('UTF-8')
    if line_str[0:11] == "Lower bound":
      temp1 = line_str.split(",")
      temp2 = temp1[0].split("=")
      lb_cost = float(temp2[1]) # lower bound from LKH.
  process.wait() # otherwise, subprocess run concurrently...

  ### get result
  res_file = problem_str+".tour"
  with open(res_file, mode="r") as fres:
    lines = fres.readlines()
    l1all = lines[1].split("=")
    tour_cost = int(l1all[1])
    ix = 6
    val = int(lines[ix])
    tour = []
    while val != -1:
      tour.append(val-1)
      # tour.append(val) # LKH node index start from 1 while cmat index in python start from 0.
      ix = ix + 1
      val = int(lines[ix])
  return tour, tour_cost, lb_cost

def invoke_concorde(exe_path, problem_str):
  """
  TODO
  """
  sys.exit("[ERROR] Not Implemented!")
  return


def reorderTour(tour, s):
  """
  find s in the tour [AAA, s, BBB] and reorder the tour to make s be the first node.
  [s,BBB,AAA]
  """
  found = 0
  for i,v in enumerate(tour):
    if v == s:
      found = 1
      break
  if not found:
    print("[ERROR] s = ", s, " is not in the tour ", tour)
    sys.exit("[ERROR] reorderTour fail to find s!")
  return tour[i:]+tour[:i]