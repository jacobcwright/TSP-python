# TSP for cs312

To start the gui and play with the TSP, type `python3 Proj5GUI.py`

3 algorithms are implemented to solve the TSP problem:

1. Default: Brute Force Algorithm
2. Greedy: Nearest Neighbor Algorithm
3. Fancy: 2-opt Algorithm

Had a near working Christofides algorithm, however, in the hard-deterministic mode the graph becomes asymmetric and directed. This caused the algorithm to fail 50% of the time when changing the eulerian tour to a hamiltonian tour.

Another approach was a similar MST heuristic. I found the MST of the cities, did a DFS on the MST, and then added the edges that were not in the MST. This was a good heuristic, but it was not as good as the 2-opt algorithm and ran into similar issues as the Christofides algorithm when using asymmetric graphs.
