# noinspection PyProtectedMember

import heapq
import itertools
import math

def calculate_modulation(distance, reach_m, max_m):
    if distance <= reach_m:
        return max_m
    elif distance > reach_m * (1 << (max_m - 1)):
        return 0
    else:
        return max_m + 1 - int(math.ceil(math.log2(2 * distance / reach_m)))

def calculate_distance(modulation, reach_m, max_m):
    return reach_m * (1 << (max_m - modulation))

def calculate_slots(demand, slot_bw, modulation):
    return int(math.ceil(demand / (slot_bw * modulation)))

def calculate_bw(slots, slot_bw, modulation):
    return slots * slot_bw * modulation

def decide_default(demand, distance, slots_available, slot_bw, reach_m, max_m):
    modulation = calculate_modulation(distance, reach_m, max_m)
    if modulation == 0:
        return False
    slots_required = calculate_slots(demand, slot_bw, modulation)
    return slots_required <= slots_available

def check_slots(graph, path, cu, what):
    cu_cmp = cu if what == 'reserve' else 0
    edges = []
    distance = 0.0
    for src, dst, key in path:
        try:
            # noinspection PyProtectedMember
            edge = graph._succ[src][dst][key]
        except KeyError:
            raise ValueError(f'Invalid path: optical edge between {src} and {dst} with key {key} not found')
        if cu & edge['au'] == cu_cmp:
            edges.append(edge)
            distance += edge['distance']
        else:
            cu_end = cu.bit_length()
            powered = (1 << cu_end) - 1
            cu_start = (cu ^ powered).bit_length()
            if what == 'reserve':
                raise ValueError(f'Cannot {what} slots {cu_start}-{cu_end}, '
                                 f'they are not free on optical edge between {src} {dst} {key}')
            else:
                raise ValueError(f'Cannot {what} slots {cu_start}-{cu_end}, '
                                 f'they are not reserved on optical edge between {src} {dst} {key}')
    return distance, edges

def xor_slots(edges, cu):
    for edge in edges:
        edge['au'] ^= cu

def iterate_continuous_blocks_from_end(slots_set):
    while slots_set:
        end = slots_set.bit_length()
        powered = (1 << end) - 1
        start = (slots_set ^ powered).bit_length()
        new_set = (powered >> start) << start
        yield (start, end, new_set)
        slots_set ^= new_set

def filter_edges(g, org_g, cu_set, weight='cost'):
    for u, du in g._succ.items():
        for v, duv in du.items():
            for k, data in duv.items():
                if cu_set & data['au'] == cu_set:
                    data[weight] = org_g._succ[u][v][k][weight]
                else:
                    data[weight] = None

# Filtered Graphs Algorithm

def backtrack(g, preds, sources, target, weight='cost'):
    paths_to_target = []

    stack = [[(target, None, 0), 0]]
    top = 0
    while top >= 0:
        (u, v, k), i = stack[top]
        if u in sources:
            paths_to_target.append([edge for edge, n in reversed(stack[1:top + 1])])
            break
        if u not in sources and u in preds and len(preds[u]) > i:
            edge = preds[u][i]
            top += 1
            if top == len(stack):
                stack.append([edge, 0])
            else:
                stack[top] = [edge, 0]
        else:
            stack[top - 1][1] += 1
            top -= 1

    return paths_to_target

def dijkstra(g, sources, target, weight='cost', preds=None, paths=None,
             cutoff=None, cu_set=None):
    """Uses Dijkstra's algorithm to find shortest weighted paths

    Parameters
    ----------
    g : NetworkX graph

    sources : non-empty iterable of nodes
        Starting nodes for paths. If this is just an iterable containing
        a single node, then all paths computed by this function will
        start from that node. If there are two or more nodes in this
        iterable, the computed paths may begin from any one of the start
        nodes.

    weight: str
        Function with (u, v, data) input that returns that edges weight

    paths: dict of lists, optional (default=None)
        empty dict to store the path list from source to each node, keyed by node.
        If None, paths are not stored.

    target : node label, optional
        Ending node for paths. Search is halted when target is found.

    cutoff : integer or float, optional
        Depth to stop the search. Only return paths with length <= cutoff.

    Returns
    -------
    distance : dictionary
        A mapping from node to shortest distance to that node from one
        of the source nodes.

    Notes
    -----
    The optional predecessor and path dictionaries can be accessed by
    the caller through the original pred and paths objects passed
    as arguments. No need to explicitly return pred or paths.

    """

    if not sources:
        raise ValueError('sources must not be empty')
    if paths:
        raise ValueError('paths must be an empty dict')

    g_succ = g._succ if g.is_directed() else g._adj

    if paths is not None:
        for source in sources:
            paths[source] = [[]]

    push = heapq.heappush
    pop = heapq.heappop
    dist = {}  # dictionary of final distances
    seen = {}
    # fringe is heapq with 3-tuples (distance,c,node)
    # use the count c to avoid comparing nodes (may not be able to)
    c = itertools.count()
    fringe = []
    for source in sources:
        seen[source] = 0
        push(fringe, (0, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break
        for u, edges in g_succ[v].items():
            for key, data in edges.items():
                if cu_set is None or cu_set & data['au'] == cu_set:
                    cost = data[weight]
                else:
                    cost = None
                if cost is None:
                    continue
                vu_dist = d + cost
                if cutoff is not None:
                    if vu_dist > cutoff:
                        continue
                # if u in dist:
                #     if vu_dist < dist[u]:
                #         raise ValueError('Contradictory paths found:',
                #                          'negative weights?')
                seen_dist = seen.get(u, math.inf)
                if vu_dist < seen_dist:
                    seen[u] = vu_dist
                    push(fringe, (vu_dist, next(c), u))
                    if preds is not None:
                        preds[u] = [(v, u, key)]
                    if paths is not None:
                        paths[u] = [[*path, (v, u, key)] for path in paths[v]]
                # elif vu_dist == seen_dist:
                #     if preds is not None:
                #         preds[u].append((v, u, key))
                #     if paths is not None:
                #         paths[u].extend([*path, (v, u, key)] for path in paths[v])

    # The optional predecessor and path dictionaries can be accessed
    # by the caller via the preds and paths objects passed as arguments.
    return dist

def dijkstra_filtered(g, sources, target, demand, max_cu, slot_bw, reach_m, max_m, weight='cost', decide=decide_default):
    best_dist = math.inf
    paths_to_target = []

    for modulation in range(max_m, 0, -1):
        slots = calculate_slots(demand, slot_bw, modulation)
        if slots > max_cu:
            continue
        max_distance = calculate_distance(modulation, reach_m, max_m)

        # if not inline:
        #     org_g = g
        #     g = org_g.copy()

        cu_set = (1 << slots) - 1
        for cu_start in range(max_cu - slots + 1):
            if cu_start:
                cu_set <<= 1
            # if not inline:
            #     filter_edges(g, org_g, cu_set, weight)
            preds = {}
            # paths = {}
            dists = dijkstra(g, sources, target, weight=weight, preds=preds, cutoff=max_distance, cu_set=cu_set)
            # if target in dists:
            #     assert paths[target] == backtrack(g, preds, sources, target, weight)
            if target in dists:
                assert decide(demand, dists[target], slots, slot_bw, reach_m, max_m)  # The same as cutoff=max_distance?
                if dists[target] < best_dist:
                    paths_to_target = [(cu_start, cu_start + slots, path) for path in backtrack(g, preds, sources, target, weight)]
                    best_dist = dists[target]

        if paths_to_target:
            return best_dist, paths_to_target

    return best_dist, paths_to_target

# Generic Dijkstra Algorithm (https://arxiv.org/abs/1810.04481)

class Label:
    __slots__ = ['cost', 'cu_start', 'cu_end', 'cu_set', 'edge']

    def __init__(self, cost, cu_start, cu_end, cu_set, edge):
        self.cost = cost
        self.cu_start = cu_start
        self.cu_end = cu_end
        self.cu_set = cu_set
        self.edge = edge

    def __str__(self):
        return f'{self.edge} {self.cu_start} {self.cu_end} {self.cost}'

    def __hash__(self):
        return hash((self.cost, self.cu_set, self.edge))

    def __lt__(self, other):
        return self.cu_start <= other.cu_start and self.cu_end >= other.cu_end and \
               (self.cost < other.cost or (self.cost == other.cost and self.cu_set != other.cu_set))

    def __le__(self, other):
        return NotImplemented

    def __eq__(self, other):
        return self.cost == other.cost and self.cu_set == other.cu_set

    def __gt__(self, other):
        return NotImplemented

    def __ge__(self, other):
        return NotImplemented

def backtrack_generic(g, labels, sources, target, weight='cost'):
    paths_to_target = []
    labels = {node: list(label_dict) for node, label_dict in labels.items()}

    stack = [[(target, None, 0), 0]]
    top = 0
    current_cu_start = None
    current_cu_end = None
    current_cost = math.inf
    while top >= 0:
        (u, v, k), i = stack[top]
        if u in sources:
            paths_to_target.append((current_cu_start, current_cu_end, [edge for edge, n in reversed(stack[1:top + 1])]))
            break
        if u not in sources and u in labels and len(labels[u]) > i:
            label = labels[u][i]
            if top == 0:
                current_cu_start = label.cu_start
                current_cu_end = label.cu_end
            else:
                if not (label.cu_start <= current_cu_start and label.cu_end >= current_cu_end) or label.cost + g._succ[u][v][k][weight] != current_cost:
                    stack[top][1] += 1
                    continue
            current_cost = label.cost
            top += 1
            if top == len(stack):
                stack.append([label.edge, 0])
            else:
                stack[top] = [label.edge, 0]
        else:
            stack[top - 1][1] += 1
            top -= 1
            if v is not None:
                current_cost += g._succ[u][v][k][weight]

    return current_cost, paths_to_target

def dijkstra_generic(g, sources, target, demand, max_cu, slot_bw, reach_m, max_m, weight='cost', decide=decide_default):
    if not sources:
        raise ValueError('sources must not be empty')

    g_succ = g._succ if g.is_directed() else g._adj

    push = heapq.heappush
    pop = heapq.heappop
    perm_labels = {}
    tent_labels = {}
    queue = []

    assert len(sources) == 1
    assert target

    for source in sources:
        lab = Label(0, 0, max_cu, (1 << max_cu) - 1, (None, source, 0))
        tent_labels[source] = {lab: True}
        perm_labels[source] = {}
        push(queue, (0, 0, lab))
    while queue:
        _, _, lab = pop(queue)
        v = lab.edge[1]
        if tent_labels[v].pop(lab, None) is None:
            continue
        perm_labels[v][lab] = True
        if v == target:
            break
        for u, edges in g_succ[v].items():
            for key, data in edges.items():
                cost = data[weight]
                if cost is None:
                    continue
                cost = lab.cost + cost
                for cu_start, cu_end, cu_set in iterate_continuous_blocks_from_end(lab.cu_set & data['au']):
                    lab_new = Label(cost, cu_start, cu_end, cu_set, (v, u, key))
                    if decide(demand, lab_new.cost, lab_new.cu_end - lab_new.cu_start, slot_bw, reach_m, max_m):
                        perm_labels_u = perm_labels.setdefault(u, {})
                        # ctn_perm[len(perm_labels_u)] += 1
                        # assert lab_new not in perm_labels_u
                        if not any(lab_u < lab_new for lab_u in perm_labels_u):
                            tent_labels_u = tent_labels.setdefault(u, {})
                            # assert len(perm_labels_u) + len(tent_labels_u) <= max_cu * len(g._pred[u]) / 2
                            # ctn_tent[len(tent_labels_u)] += 1
                            # assert lab_new not in tent_labels_u  # Excludes non-euclidean costs
                            # if lab_new not in tent_labels_u:  # Speeds up graphs with many shortest paths
                            to_del = []
                            for lab_u in tent_labels_u:
                                # Merge to single iteration (modification over the arxiv paper)
                                if lab_u < lab_new:
                                    break
                                if lab_new < lab_u:
                                    to_del.append(lab_u)
                            else:
                                for lab_u in to_del:
                                    del tent_labels_u[lab_u]
                                tent_labels_u[lab_new] = True
                                push(queue, (lab_new.cost, lab_new.cu_start, lab_new))

    return backtrack_generic(g, perm_labels, sources, target, weight)
