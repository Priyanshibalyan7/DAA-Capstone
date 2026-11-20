"""
Mini-Courier Planner  -  DAA Capstone (ETCCDA302)

Integrates four algorithm families into one pipeline:
  1. Sorting              -> Merge Sort prioritises parcels (deadline, then priority)
  2. Dynamic Programming  -> 0/1 Knapsack picks the best set of parcels for one trip
  3. Graph algorithm      -> Dijkstra gives shortest paths between delivery points
  4. Approximation        -> Nearest-Neighbour TSP orders the stops of the route

Usage:
    python courier_planner.py                       # run the demo on data/small_dataset.json
    python courier_planner.py --data data/small_dataset.json --capacity 20
    python courier_planner.py --benchmark           # runtime study + charts
"""
import argparse
import heapq
import itertools
import json
import random
import time
from pathlib import Path

INF = float("inf")
DEPOT = "Depot"


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
class Parcel:
    """A parcel: id, destination node, weight (kg), priority value, deadline (hours)."""

    def __init__(self, pid, dest, weight, priority, deadline):
        self.pid, self.dest = pid, dest
        self.weight, self.priority, self.deadline = weight, priority, deadline

    def __repr__(self):
        return (f"{self.pid}(dest={self.dest}, w={self.weight}kg, "
                f"prio={self.priority}, due={self.deadline}h)")


def load_dataset(path):
    """JSON format: {"graph": {"A": {"B": 4, ...}, ...}, "parcels": [{...}, ...]}"""
    with open(path) as f:
        data = json.load(f)
    graph = {u: dict(nb) for u, nb in data["graph"].items()}
    # make roads undirected and make sure every node has an entry
    for u, nb in list(graph.items()):
        for v, w in nb.items():
            graph.setdefault(v, {})
            graph[v].setdefault(u, w)
    parcels = [Parcel(p["id"], p["dest"], p["weight"], p["priority"], p["deadline"])
               for p in data["parcels"]]
    return graph, parcels


# --------------------------------------------------------------------------
# PHASE 2a - Sorting : Merge Sort  O(n log n) time, O(n) space, stable
# --------------------------------------------------------------------------
def sort_key(p):
    # earliest deadline first; ties -> higher priority first
    return (p.deadline, -p.priority)


def merge_sort(items, key=sort_key):
    if len(items) <= 1:
        return items[:]
    mid = len(items) // 2
    left, right = merge_sort(items[:mid], key), merge_sort(items[mid:], key)
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):      # <= keeps the sort stable
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:]); out.extend(right[j:])
    return out


def insertion_sort(items, key=sort_key):
    """O(n^2) baseline used only for the runtime comparison."""
    a = items[:]
    for i in range(1, len(a)):
        cur, j = a[i], i - 1
        while j >= 0 and key(a[j]) > key(cur):
            a[j + 1] = a[j]; j -= 1
        a[j + 1] = cur
    return a


# --------------------------------------------------------------------------
# PHASE 2b - Dynamic Programming : 0/1 Knapsack  O(n*W) time and space
# --------------------------------------------------------------------------
def knapsack(parcels, capacity):
    """Pick the subset of parcels with maximum total priority within the weight capacity.
    Weights must be integers. Returns (best_value, chosen_parcels, dp_table)."""
    n = len(parcels)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        p = parcels[i - 1]
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]                       # skip parcel i
            if p.weight <= w:                             # or take it
                dp[i][w] = max(dp[i][w], dp[i - 1][w - p.weight] + p.priority)
    # back-track through the table to recover which parcels were taken
    chosen, w = [], capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen.append(parcels[i - 1]); w -= parcels[i - 1].weight
    chosen.reverse()
    return dp[n][capacity], chosen, dp


def knapsack_brute_force(parcels, capacity):
    """O(2^n) checker used to prove the DP answer is optimal on small inputs."""
    best = 0
    for r in range(len(parcels) + 1):
        for combo in itertools.combinations(parcels, r):
            if sum(p.weight for p in combo) <= capacity:
                best = max(best, sum(p.priority for p in combo))
    return best


# --------------------------------------------------------------------------
# PHASE 3a - Graph : Dijkstra with a binary heap  O((V+E) log V)
# --------------------------------------------------------------------------
def dijkstra(graph, src):
    dist = {v: INF for v in graph}
    prev = {v: None for v in graph}
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue                                      # stale entry
        for v, w in graph[u].items():
            if d + w < dist[v]:
                dist[v], prev[v] = d + w, u
                heapq.heappush(heap, (dist[v], v))
    return dist, prev


def path_to(prev, target):
    path = []
    while target is not None:
        path.append(target); target = prev[target]
    return path[::-1]


# --------------------------------------------------------------------------
# PHASE 3b - Approximation : Nearest-Neighbour TSP heuristic  O(k^2) on k stops
# --------------------------------------------------------------------------
def nearest_neighbour_route(graph, stops, start=DEPOT):
    """Build a closed tour start -> ... -> start over `stops`, always going to the
    closest unvisited stop (distance = Dijkstra shortest path).
    Returns (tour_order, total_length, full_node_path)."""
    sp = {s: dijkstra(graph, s) for s in set(stops) | {start}}   # one Dijkstra per stop
    unvisited, tour, cur, total = set(stops) - {start}, [start], start, 0
    while unvisited:
        nxt = min(unvisited, key=lambda s: sp[cur][0][s])
        total += sp[cur][0][nxt]
        tour.append(nxt); unvisited.remove(nxt); cur = nxt
    total += sp[cur][0][start]
    tour.append(start)
    # expand each leg into the real road path
    full = [start]
    for a, b in zip(tour, tour[1:]):
        full += path_to(sp[a][1], b)[1:]                 # drop duplicate first node
    return tour, total, full


def tsp_optimal(graph, stops, start=DEPOT):
    """Brute-force optimum, only for tiny inputs - used to measure the heuristic's quality."""
    stops = list(set(stops) - {start})
    sp = {s: dijkstra(graph, s)[0] for s in stops + [start]}
    best = INF
    for perm in itertools.permutations(stops):
        order = [start, *perm, start]
        best = min(best, sum(sp[a][b] for a, b in zip(order, order[1:])))
    return best


# --------------------------------------------------------------------------
# Integrated pipeline
# --------------------------------------------------------------------------
def timed(fn, *a, **kw):
    t = time.perf_counter()
    r = fn(*a, **kw)
    return r, (time.perf_counter() - t) * 1000          # milliseconds


def plan_trip(graph, parcels, capacity, verbose=True):
    """Sort -> Knapsack -> Dijkstra/NN-TSP. Returns a result dict incl. step timings."""
    out = {}
    sorted_p, out["t_sort"] = timed(merge_sort, parcels)
    (best, chosen, dp), out["t_knap"] = timed(knapsack, sorted_p, capacity)
    stops = sorted({p.dest for p in chosen})
    (tour, length, full), out["t_route"] = timed(nearest_neighbour_route, graph, stops)
    out.update(sorted=sorted_p, value=best, chosen=chosen, dp=dp,
               tour=tour, length=length, full_path=full,
               left_over=[p for p in sorted_p if p not in chosen])
    if verbose:
        report(out, capacity)
    return out


def report(r, capacity):
    line = "=" * 70
    print(line + "\n STEP 1  Parcels sorted by deadline, then priority (Merge Sort)\n" + line)
    print(f" {'Rank':<5}{'Parcel':<8}{'Dest':<8}{'Wt(kg)':<8}{'Prio':<6}{'Due(h)'}")
    for i, p in enumerate(r["sorted"], 1):
        print(f" {i:<5}{p.pid:<8}{p.dest:<8}{p.weight:<8}{p.priority:<6}{p.deadline}")
    print("\n" + line + f"\n STEP 2  Knapsack packing plan (van capacity = {capacity} kg)\n" + line)
    print(f" Parcels loaded : {', '.join(p.pid for p in r['chosen'])}")
    print(f" Total weight   : {sum(p.weight for p in r['chosen'])} kg / {capacity} kg")
    print(f" Total priority : {r['value']}")
    print(f" Left for later : {', '.join(p.pid for p in r['left_over']) or '-'}")
    print("\n" + line + "\n STEP 3  Delivery route (Dijkstra + Nearest-Neighbour TSP)\n" + line)
    print(" Stop order     : " + " -> ".join(r["tour"]))
    print(" Road path      : " + " -> ".join(r["full_path"]))
    print(f" Route length   : {r['length']} km")
    print("\n" + line + "\n RUNTIME\n" + line)
    print(f" Sort {r['t_sort']:.4f} ms | Knapsack {r['t_knap']:.4f} ms | Route {r['t_route']:.4f} ms")


# --------------------------------------------------------------------------
# PHASE 4 - Testing and benchmarking
# --------------------------------------------------------------------------
def random_instance(n_parcels, n_nodes, seed=1):
    """Random connected road network + parcels (used for large datasets)."""
    rnd = random.Random(seed)
    nodes = [DEPOT] + [f"N{i}" for i in range(1, n_nodes)]
    graph = {v: {} for v in nodes}
    for i in range(1, n_nodes):                               # spanning tree => connected
        j = rnd.randrange(i)
        w = rnd.randint(1, 20)
        graph[nodes[i]][nodes[j]] = graph[nodes[j]][nodes[i]] = w
    for _ in range(n_nodes * 2):                              # extra roads
        a, b = rnd.sample(nodes, 2)
        w = rnd.randint(1, 20)
        if b not in graph[a] or w < graph[a][b]:
            graph[a][b] = graph[b][a] = w
    parcels = [Parcel(f"P{i}", rnd.choice(nodes[1:]), rnd.randint(1, 15),
                      rnd.randint(1, 50), rnd.randint(1, 72)) for i in range(n_parcels)]
    return graph, parcels


def run_tests():
    print("Correctness tests")
    print("-" * 40)
    # 1. sorting
    _, ps = random_instance(200, 30, seed=7)
    assert merge_sort(ps) == sorted(ps, key=sort_key) == insertion_sort(ps)
    print(" [PASS] merge sort == insertion sort == Python sorted (200 parcels)")
    # 2. knapsack vs brute force on many random small cases
    for s in range(30):
        _, ps = random_instance(10, 8, seed=s)
        v, chosen, _ = knapsack(ps, 40)
        assert v == knapsack_brute_force(ps, 40) == sum(p.priority for p in chosen)
        assert sum(p.weight for p in chosen) <= 40
    print(" [PASS] knapsack == brute force optimum (30 random cases, 10 parcels)")
    # 3. dijkstra on hand-made graph
    g = {"A": {"B": 1, "C": 4}, "B": {"A": 1, "C": 2, "D": 6},
         "C": {"A": 4, "B": 2, "D": 3}, "D": {"B": 6, "C": 3}}
    d, pr = dijkstra(g, "A")
    assert d == {"A": 0, "B": 1, "C": 3, "D": 6} and path_to(pr, "D") == ["A", "B", "C", "D"]
    print(" [PASS] Dijkstra distances/paths on a hand-checked graph")
    # 4. NN tour is valid and never shorter than the optimum; report ratio
    ratios = []
    for s in range(30):
        g, ps = random_instance(8, 12, seed=s)
        stops = sorted({p.dest for p in ps})[:7]
        _, length, full = nearest_neighbour_route(g, stops)
        opt = tsp_optimal(g, stops)
        assert length >= opt and full[0] == full[-1] == DEPOT
        ratios.append(length / opt)
    print(f" [PASS] NN-TSP valid closed tour; length/optimal: avg {sum(ratios)/len(ratios):.3f}, "
          f"worst {max(ratios):.3f} (30 cases, 7 stops)")
    print()


def benchmark(outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(exist_ok=True)

    # (a) sorting: merge sort vs insertion sort
    sizes = [100, 200, 400, 800, 1600, 3200]
    ms, ins = [], []
    for n in sizes:
        _, ps = random_instance(n, 20, seed=n)
        ms.append(timed(merge_sort, ps)[1]); ins.append(timed(insertion_sort, ps)[1])
    # (b) knapsack: scales with n*W
    ks = []
    ksizes = [50, 100, 200, 400, 800]
    for n in ksizes:
        _, ps = random_instance(n, 20, seed=n)
        ks.append(timed(knapsack, ps, 500)[1])
    # (c) Dijkstra on growing graphs
    vsizes = [100, 200, 400, 800, 1600, 3200]
    dj = []
    for v in vsizes:
        g, _ = random_instance(1, v, seed=v)
        dj.append(timed(dijkstra, g, DEPOT)[1])
    # (d) NN-TSP on growing number of stops (graph fixed at 400 nodes)
    g, _ = random_instance(1, 400, seed=5)
    nodes = [v for v in g if v != DEPOT]
    tsizes = [5, 10, 20, 40, 80]
    nn = [timed(nearest_neighbour_route, g, nodes[:k])[1] for k in tsizes]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0][0].plot(sizes, ins, "o-", label="Insertion sort O(n²)")
    ax[0][0].plot(sizes, ms, "s-", label="Merge sort O(n log n)")
    ax[0][0].set(title="Sorting parcels", xlabel="parcels (n)", ylabel="ms"); ax[0][0].legend()
    ax[0][1].plot(ksizes, ks, "o-", color="tab:green")
    ax[0][1].set(title="0/1 Knapsack DP (W = 500)", xlabel="parcels (n)", ylabel="ms")
    ax[1][0].plot(vsizes, dj, "o-", color="tab:red")
    ax[1][0].set(title="Dijkstra (heap)", xlabel="nodes (V)", ylabel="ms")
    ax[1][1].plot(tsizes, nn, "o-", color="tab:purple")
    ax[1][1].set(title="Nearest-Neighbour TSP (k Dijkstra runs)", xlabel="stops (k)", ylabel="ms")
    for a in ax.flat:
        a.grid(alpha=.3)
    fig.suptitle("Mini-Courier Planner - runtime of major steps")
    fig.tight_layout(); fig.savefig(outdir / "runtime_charts.png", dpi=130); plt.close(fig)

    # (e) whole pipeline, small vs large dataset -> stacked bar
    labels, parts = [], {"Sort": [], "Knapsack": [], "Route": []}
    print(f"{'Dataset':<22}{'Sort ms':>10}{'Knap ms':>10}{'Route ms':>10}{'Length':>8}")
    gs, ps_ = load_dataset(Path(__file__).parent / "data" / "small_dataset.json")
    cases = [("small (10 parcels)", gs, ps_, 20)]
    for n, v, cap in [(100, 100, 100), (500, 300, 300), (1000, 500, 500)]:
        g, p = random_instance(n, v, seed=n)
        cases.append((f"{n} parcels/{v} nodes", g, p, cap))
    for name, g, p, cap in cases:
        r = plan_trip(g, p, cap, verbose=False)
        labels.append(name)
        for k, t in zip(parts, ("t_sort", "t_knap", "t_route")):
            parts[k].append(r[t])
        print(f"{name:<22}{r['t_sort']:>10.3f}{r['t_knap']:>10.3f}{r['t_route']:>10.3f}{r['length']:>8}")
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = [0] * len(labels)
    for k, vals in parts.items():
        ax.bar(labels, vals, bottom=bottom, label=k)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set(title="Pipeline runtime by dataset size", ylabel="ms"); ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "pipeline_runtime.png", dpi=130); plt.close(fig)
    print(f"\nCharts saved to {outdir}/")


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Mini-Courier Planner")
    ap.add_argument("--data", default=str(here / "data" / "small_dataset.json"))
    ap.add_argument("--capacity", type=int, default=20, help="van weight capacity (kg)")
    ap.add_argument("--benchmark", action="store_true", help="run tests + runtime study")
    args = ap.parse_args()
    if args.benchmark:
        run_tests()
        benchmark(here / "outputs")
    else:
        graph, parcels = load_dataset(args.data)
        plan_trip(graph, parcels, args.capacity)


if __name__ == "__main__":
    main()
