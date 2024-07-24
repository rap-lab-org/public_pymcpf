# Multi-Agent Combinatorial Path Finding

This proejct is about Multi-Agent Combinatorial Path Finding (MCPF). The goal is to compute collision-free paths for multiple agents from their starts to destinations while visiting a large number of intermediate target locations along the paths. Intuitively, MCPF is a combination of mTSP (multiple traveling salesman problem) and MAPF (multi-agent path finding). MCPF also involves assignment constraints, which specify the subsets of agents that are eligible to visit each target/destination. This repo provides a python implementation of the Conflict-Based Steiner Search (CBSS) algorithm which solves MCPF. More technical details can be found in the [paper](http://www.roboticsproceedings.org/rss18/p058.pdf), [video](https://youtu.be/xwLoCiJ2vJY) or [contact](https://wonderren.github.io/).

<p align="center">
<img src="https://github.com/wonderren/wonderren.github.io/blob/master/images/fig_cbss_random.gif" alt="" hspace="15" style=" border: #FFFFFF 2px none;">
</p>

(Fig 1: A conflict-free joint path to a MCPF problem instance. Triangles are intermediate targets while stars are destinations. For the assignment constraints: each agent has a pre-assigned destination and one pre-assigned target (indicated by the color), while all remaining targets are fully anonymous (blue triangles).)

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

### About the Low-Level Search

The original implementation has a flaw in its low-level search, which runs sequential A\* and ignores the influence across targets. As a result, it may not return an optimal individual path. The latest version (tag: v1.1 and thereafter) has fixes this issue on the low-level search.

### About Solution Optimality

CBSS is theoretically guaranteed to find an optimal or bounded sub-optimal solution joint path, when the underlying TSP solver is guaranteed to solve TSPs to optimality.
The current implementation of CBSS depends on LKH, which is a popular heuristic algorithm that is not guaranteed to find an optimal solution to TSP. Therefore, the resulting CBSS implementation is not guaranteed to return an optimal solution.
However, LKH has been shown to return an optimal solution for numerous TSP instances in practice, this implementation of CBSS should also be able to provide optimal solutions for many MCPF instances.
If the optimality of the solution must be guaranteed, one can consider leveraging the [Concorde](https://www.math.uwaterloo.ca/tsp/concorde.html) TSP solver (or other TSP solvers that can guarantee solution optimality) to replace LKH.

### Related Papers

[1] Ren, Zhongqiang, Sivakumar Rathinam, and Howie Choset. "CBSS: A New Approach for Multiagent Combinatorial Path Finding." IEEE Transaction on Robotics (T-RO), 2023.\
[[Bibtex](https://wonderren.github.io/files/bibtex_ren23cbssTRO.txt)]
[[Paper](https://wonderren.github.io/files/ren23_CBSS_TRO.pdf)]
[[Talk](https://youtu.be/V17vQSZP5Zs?t=2853)]

[2] Ren, Zhongqiang, Sivakumar Rathinam, and Howie Choset. "MS*: A new exact algorithm for multi-agent simultaneous multi-goal sequencing and path finding." In 2021 IEEE International Conference on Robotics and Automation (ICRA), pp. 11560-11565. IEEE, 2021.\
[[Bibtex](https://wonderren.github.io/files/bibtex_ren21ms.txt)]
[[Paper](https://wonderren.github.io/files/ren21-MSstar.pdf)]
[[Talk](https://youtu.be/cjwO4yycfpo)]
