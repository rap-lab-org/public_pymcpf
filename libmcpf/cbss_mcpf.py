"""
Author: Zhongqiang (Richard) Ren
Version@2021-07
All Rights Reserved
ABOUT: this file constains CBSS-MCPF-AC, which is derived from CBSS (framework) and aim to solve MCPF-AC problems.
"""

import cbss
import seq_mcpf

class CbssMCPF(cbss.CbssFramework) :
  """
  """
  def __init__(self, grids, starts, goals, dests, ac_dict, configs):
    """
    """
    # mtsp_solver = msmp_seq.BridgeLKH_MSMP(grids, starts, goals, dests)
    # mtsp_solver = mcpf_seq.BridgeLKH_MCPF(grids, starts, goals, dests, ac_dict) # NOTE that ac_dict is only used in mtsp_solver, not in CBSS itself.
    mtsp_solver = seq_mcpf.SeqMCPF(grids, starts, goals, dests, ac_dict, configs) # NOTE that ac_dict is only used in mtsp_solver, not in CBSS itself.
    super(CbssMCPF, self).__init__(mtsp_solver, grids, starts, goals, dests, dict(), configs)
    return

def RunCbssMCPF(grids, starts, targets, dests, ac_dict, configs):
  """
  starts, targets and dests are all node ID.
  heu_weight and prune_delta are not in use. @2021-05-26
  """
  ccbs_planner = CbssMCPF(grids, starts, targets, dests, ac_dict, configs)
  path_set, search_res = ccbs_planner.Search_ml()
  # print(path_set)
  # print(res_dict)
  res_dict = dict()
  res_dict["path_set"] = path_set
  res_dict["round"] = search_res[0] # = num of high level nodes closed.
  res_dict["best_g_value"] = search_res[1]
  res_dict["open_list_size"] = search_res[3]
  res_dict["num_low_level_expanded"] = search_res[4]
  res_dict["search_success"] = search_res[5]
  res_dict["search_time"] = search_res[6]
  res_dict["n_tsp_call"] = search_res[7]
  res_dict["n_tsp_time"] = search_res[8]
  res_dict["n_roots"] = search_res[9]

  return res_dict
