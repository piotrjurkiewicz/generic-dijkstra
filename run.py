import itertools
import json
import random
import sys
import time

import networkx as nx

import eon

if hasattr(time, 'process_time_ns'):
    process_time = time.process_time_ns
else:
    process_time = time.process_time

def run(algorithm, seed, weight='distance', log=1):
    if algorithm == 'filtered':
        dijkstra = eon.dijkstra_filtered
    elif algorithm == 'generic':
        dijkstra = eon.dijkstra_generic
    else:
        raise ValueError

    slot_bw = 1
    max_m = 4

    print('seed', 'nodes', 'topo_num', 'edges', 'units', 'mean_demand', 'n', 'bad', 'cum_demand', 'cum_util', 'src', 'dst', 'demand', 'paths', 'path_len', 'cu_start', 'elapsed')

    for mean_demand_fraction in [10, 20]:
        for nodes in range(25, 275, 25):
            for topo_num in range(0, 10):
                for units in range(100, 1100, 100):
                    mean_demand = units // mean_demand_fraction
                    rng_demand = random.Random(seed)
                    rng_path_choice = random.Random(seed)
                    topo_name = 'gabriel/%s/%s' % (nodes, topo_num)
                    g = nx.node_link_graph(json.load(open('topo_lib/%s.json' % topo_name)))
                    if not isinstance(g, nx.DiGraph):
                        g = g.to_directed()
                    edges = 0
                    for edge in g.edges.values():
                        edge['au'] = (1 << units) - 1
                        edge['cost'] = 1
                        edges += 1
                    demands_keys = []
                    demands_values = []
                    if 'demands' in g.graph:
                        for src, demands in g.graph['demands'].items():
                            for dst, demand in demands.items():
                                demands_keys.append((src, dst))
                                demands_values.append(demand)
                    if not demands_keys:
                        demands_keys = [pair for pair in itertools.permutations(g, 2)]
                    n = 0
                    bad = 0
                    cum_time = 0
                    cum_demand = 0
                    cum_util = 0
                    reach_1 = 0
                    for line in open('topo_lib/%s.csv' % topo_name):
                        try:
                            sp_length = float(line.split(',')[-4])
                        except Exception:
                            continue
                        reach_1 = max(reach_1, sp_length * 1.5)
                    reach_m = reach_1 / 2 ** (max_m - 1)
                    while True:
                        if demands_values:
                            src, dst = rng_demand.choices(demands_keys, demands_values)[0]
                        else:
                            src, dst = rng_demand.choice(demands_keys)
                        demand = rng_demand.randint(1, mean_demand * 2 - 1)
                        start_time = process_time()
                        paths = dijkstra(g, {src}, dst, demand, units, slot_bw, reach_m, max_m, weight=weight)
                        elapsed = process_time() - start_time
                        if isinstance(elapsed, float):
                            elapsed = int(elapsed * 1000000000)
                        if paths[1]:
                            cu_start, cu_end, path = rng_path_choice.choice(sorted(paths[1]))
                            cu_set = ((1 << demand) - 1) << cu_start
                            if log > 2:
                                print((cu_start, cu_start + demand), path)

                            for u, v, key in path:
                                # noinspection PyProtectedMember
                                g._succ[u][v][key]['au'] ^= cu_set
                        else:
                            cu_start, path = 0, []
                        if log:
                            print(seed, nodes, topo_num, edges, units, mean_demand, n, bad, cum_demand, cum_util, src, dst, demand, len(paths), len(path), cu_start, elapsed)
                        n += 1
                        cum_time += elapsed
                        if path:
                            cum_demand += demand
                            cum_util += demand * len(path)
                        else:
                            bad += 1
                            if cum_util / (edges * units) > 0.6:
                                break
                    if log == 0:
                        print(seed, nodes, topo_num, edges, units, mean_demand, n, bad, cum_demand, cum_util / (edges * units), '-', '-', '-', '-', '-', '-', cum_time)
                    # exit()


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]))
