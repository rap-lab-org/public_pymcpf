# Multi-Agent Combinatorial Path Finding

This proejct is about Multi-Agent Combinatorial Path Finding (MCPF). The goal is to compute collision-free paths for multiple agents from their starts to destinations while visiting a large number of intermediate target locations along the paths. Intuitively, MCPF is a combination of mTSP (multiple traveling salesman problem) and MAPF (multi-agent path finding). MCPF also involves assignment constraints, which specify the subsets of agents that are eligible to visit each target/destination. This repo provides a python implementation of the Conflict-Based Steiner Search (CBSS) algorithm which solves MCPF. More technical details can be found in the paper (to be added).

<center><img src="https://github.com/wonderren/wonderren.github.io/blob/master/images/fig_cbss_random.gif" alt="" align="middle" hspace="15" style=" border: #FFFFFF 2px none;"></center>

(Fig 1: A conflict-free joint path to a MCPF problem instance. Triangles are intermediate targets while stars are destinations. For the assignment constraints: each agent has a pre-assigned destination and one pre-assigned target, while all remaining targets are fully anonymous.)

The code is distributed for academic and non-commercial use.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Requirements

* We use Python (3.8.10) and Ubuntu 20.04. Lower or higher version may also work.
* [LKH-2.0.9](http://webhotel4.ruc.dk/~keld/research/LKH/) is required as the underlying TSP solver. The executable of LKH should be placed at location: pytspbridge/tsp_solver/LKH-2.0.9/LKH. In other words, run `pytspbridge/tsp_solver/LKH-2.0.9/LKH` command in the terminal should be able to invoke LKH.

## Instructions:

* Run `python3 run_example_cbss.py` to run the example, which shows how to use the code.

## Others

### About Solution Optimality

CBSS is theoretically guaranteed to find optimal or bounded sub-optimal solution joint path, when the underlying TSP solver is guaranteed to solve TSP to optimality.
The current implementation of CBSS depends on LKH, which is a popular heuristic algorithm that is not guaranteed to find an optimal solution to TSP. Therefore, the resulting CBSS implementation is not guaranteed to always return an optimal solution.
However, LKH has been shown to return an optimal solution for numerous TSP instances in practice, this implementation of CBSS should also be able to provide optimal solution for many MCPF instances.
If an optimal solution to MCPF must be guaranteed anyway, one can consider leveraging the [Concorde](https://www.math.uwaterloo.ca/tsp/concorde.html) TSP solver to replace LKH.

### References

TODO,
