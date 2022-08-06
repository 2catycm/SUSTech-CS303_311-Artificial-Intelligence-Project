import argparse
import time
import random
import numpy as np
import re
import copy


def floyd(N, G):
    """
    解决多源最短路问题的弗洛伊德算法。
    :param N: 边的数量。默认以1开始编号节点。
    :param G: 邻接表形式的图，默认以0不启用。
    :return distances: distances[i, j] 表示i为起点到j的最短路 。 应当为对称矩阵
    :return paths: path[i, j] 表示i为起点，到达j的最短路上，最后一个中转节点。
    """
    distances = np.ones((N + 1, N + 1)) * np.inf  # # distances[i, j] 表示i为起点到j的最短路 。 应当为对称矩阵
    distances[np.diag_indices_from(distances)] = 0  # 自己到自己距离为0
    paths = np.ones((N + 1, N + 1)) * (-1)  # path[i, j] 表示i为起点，到达j的最短路上，最后一个中转节点。
    # 初始化一个直接连边最短路
    for i in range(1, N + 1):
        for edge in G[i]:
            distances[i, edge[0]] = edge[1]
            paths[i, edge[0]] = i  # 经过零次中转即可。
    # 算法正式开始
    for k in range(1, N + 1):  # 中间点
        for i in range(1, N + 1):  # 起点
            for j in range(1, N + 1):  # 终点
                new_cost = distances[i, k] + distances[k, j]
                if new_cost < distances[i, j]:
                    distances[i, j] = new_cost
                    paths[i, j] = k  # 中转节点

    return distances, paths


class CarpInstance:
    def __init__(self):
        self.name = "default"
        self.vertices = 0
        self.depot = 0
        self.required_edges = 0
        self.non_required_edges = 0
        self.vehicles = 0
        self.capacity = 0
        self.total_cost_of_required_edges = 0
        self.graph = None  # 每一个是邻接表。邻居的表示是 other, cost, demand. 为了效率不做面向对象。
        self.task_edges = []
        self.distances = None

    def with_file(self, filename: str):
        with open(filename) as file:
            lines = file.readlines()
        self.name = re.split("\\s+", lines[0])[-2]
        self.vertices = int(re.split("\\s+", lines[1])[-2])
        self.depot = int(re.split("\\s+", lines[2])[-2])
        self.required_edges = int(re.split("\\s+", lines[3])[-2])
        self.non_required_edges = int(re.split("\\s+", lines[4])[-2])
        self.vehicles = int(re.split("\\s+", lines[5])[-2])
        self.capacity = int(re.split("\\s+", lines[6])[-2])
        self.total_cost_of_required_edges = int(re.split("\\s+", lines[7])[-2])
        # self.graph = [[]] * (self.vertices + 1) # 不能这样写，这样 地址是一样的。
        self.graph = [[] for i in range(self.vertices + 1)]

        for line in lines[9:]:  # 直接开始读数据
            if line.startswith("END"):
                break
            elements = list(map(int, line.split()))
            self.graph[elements[0]] += [[elements[1], *elements[2:]]]
            self.graph[elements[1]] += [[elements[0], *elements[2:]]]
            if elements[3] != 0:
                self.task_edges.append(elements)
        return self

    def with_distances_calculated(self):
        assert self.graph is not None
        self.distances, _ = floyd(self.vertices, self.graph)  # 暂时不关心 paths
        return self

    def __str__(self):
        last_endl = 0
        s = "carp_instance("
        for name, value in vars(self).items():
            if name == 'graph': continue
            s += f"{name}={value}, "
            if len(s) - last_endl >= 100:
                last_endl = len(s)
                s += "\n"
        s += "\ngraph=[adj[edge[other, cost, demand], edge[...]], adj[...]]\n"
        for v in range(1, self.vertices + 1):
            s += f"adj({v}): {self.graph[v]}\n"
        return s + ")"


class CarpSolution:
    def __init__(self, routes, costs):
        self.routes = routes
        self.costs = costs

    def __str__(self):
        routes = self.routes
        costs = self.costs
        res = "s "
        for route in routes:
            res += "0,"
            for task_edge in route:
                res += f"({task_edge[0]},{task_edge[1]}),"
            res += "0,"
        res = res[:len(res) - 1]  # 多余逗号
        return res + f"\nq {int(costs):d}"


class PathScanningHeuristic:
    def __init__(self, carp_instance):
        self.carp_instance = carp_instance

    def betterThan(self, new_edge, old_edge, load, capacity, depot, distances):
        diff_ratio = new_edge[3] / new_edge[2] - old_edge[3] / old_edge[2]
        diff_distance = distances[new_edge[0], depot] - distances[old_edge[0], depot]
        full_ratio = load / capacity
        assert 0 <= full_ratio <= 1
        return diff_distance if full_ratio > 0.5 else -diff_distance

    def path_scanning(self):
        distances = self.carp_instance.distances
        unserviced = copy.deepcopy(self.carp_instance.task_edges)
        depot = self.carp_instance.depot
        capacity = self.carp_instance.capacity

        routes = []
        costs = 0
        while len(unserviced) != 0:
            route = []
            load = 0
            cost = 0
            current_end = depot
            while True:
                d = np.inf
                u = None
                u_remove = None
                for task_edge in unserviced:
                    if load + task_edge[3] > capacity:  # 3 是 demand
                        continue
                    for s, e in [[0, 1], [1, 0]]:
                        new_dist = distances[current_end][task_edge[s]]
                        new_edge = [task_edge[s], task_edge[e], task_edge[2], task_edge[3]]
                        if new_dist < d:
                            d = new_dist
                            u = new_edge
                            u_remove = task_edge
                        elif new_dist == d and self.betterThan(new_edge, u, load, capacity, depot, distances):
                            u = new_edge
                            u_remove = task_edge
                if d == np.inf:
                    break
                load += u[3]  # 3是demand
                cost += d + u[2]  # 2 是cost
                current_end = u[1]
                route.append(u)
                unserviced.remove(u_remove)
                if len(unserviced) == 0:
                    break
            cost += distances[current_end, depot]  # 回到起点
            routes.append(route)
            costs += cost
        return CarpSolution(routes, costs)


if __name__ == '__main__':
    # 创建解析步骤
    parser = argparse.ArgumentParser(description='容量限制约束弧路径问题求解器。', epilog='So what can I help you? ')

    # 添加参数步骤
    parser.add_argument('carp_instance', action='store', type=str,
                        help='the absolute path of the test CARP instance file. ',
                        )

    parser.add_argument('-t', dest='termination', action='store', type=float,
                        default=60.0,
                        help='the termination condition of the algorithm. Specifically, <termination> is a positive '
                             'number which indicates how many seconds (in Wall clock time, range: [60s, 600s]) the '
                             'algorithm can spend on this instance. Once the time budget is consumed, the algorithm '
                             'should be terminated immediately.')
    parser.add_argument('-s', dest='random_seed', action='store', type=int,
                        default=0,
                        help='the random seed used in this run. In case that your solver is stochastic, the random '
                             'seed controls all the stochastic behaviors of yoursolver, such that the same random '
                             'seeds will make your solver produce the same results. If your solver is deterministic, '
                             'it still needs to accept −s <random_seed>, but can just ignore them while solving CARPs')
    parser.add_argument('--version', action='version', version='version 0.1.0')
    # 解析参数步骤
    args = parser.parse_args()
    # print(type(args.carp_instance))
    carp_instance = CarpInstance().with_file(args.carp_instance).with_distances_calculated()
    # print(carp_instance)
    carp_solver = PathScanningHeuristic(carp_instance)
    solution = carp_solver.path_scanning()
    print(solution)
