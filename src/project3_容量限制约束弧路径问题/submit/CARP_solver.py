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
    def __init__(self, name="default", vertices=0, depot=0, required_edges=0, non_required_edges=0,
                 vehicles=0, capacity=0, total_cost_of_required_edges=0, graph=None,
                 task_edges=None, distances=None):
        if task_edges is None:
            task_edges = []
        self.name = name
        self.vertices = vertices
        self.depot = depot
        self.required_edges = required_edges
        self.non_required_edges = non_required_edges
        self.vehicles = vehicles
        self.capacity = capacity
        self.total_cost_of_required_edges = total_cost_of_required_edges
        self.graph = graph  # 每一个是邻接表。邻居的表示是 other, cost, demand. 为了效率不做面向对象。
        self.task_edges = task_edges
        self.distances = distances

    def copy(self):
        """
        复制一份自身，以便修改而不影响原始问题。
        比如需要得到子图；
        比如需要临时修改容量后跑 Path Scanning，以便得到一个 giant solution
        :return:
        """
        return CarpInstance(self.name, self.vertices, self.depot, self.required_edges, self.non_required_edges,
                            self.vehicles, self.capacity
                            , self.total_cost_of_required_edges,
                            copy.deepcopy(self.graph), copy.deepcopy(self.task_edges), copy.deepcopy(self.distances))

    def with_file(self, filename: str):
        """
        遵循数据集的格式要求，读取carp问题。
        :param filename:
        :return:
        """
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

    def costs_of(self, routes):
        depot = self.depot
        distances = self.distances
        costs = 0
        for route in routes:
            cost = 0
            current = depot
            for task_edge in route:
                cost += distances[current, task_edge[0]]  # task_edge.start
                cost += task_edge[2]  # task_edge.cost
                current = task_edge[1]  # task_edge.end
            cost += distances[current, depot]
            costs += cost
        return costs

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
    """
    简单的entity，提供 str 变换
    """

    def __init__(self, routes, costs=None):
        self.routes = routes
        self.costs = costs

    def with_costs_calculated(self, carp_instance):
        self.costs = carp_instance.costs_of(self.routes)

    def __str__(self):
        """
        按照 OJ 格式要求输出解。
        :return:
        """
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


class SolutionOperators:
    """
    对 Carp 解进行修改，以便优化算法可以优化。
    """

    def __init__(self, carp_instance):
        self.carp_instance = carp_instance

    def routes2task_edges(self, routes):
        """
        从多个 route 的集合 合并(merge) 为一条 "giant route" （不符合容量约束）
        需要通过 @task_edges2routes() 方法重新获得合法的 routes 集合。
        :param routes:
        :return:
        """
        pass

    def task_edges2routes(self):
        pass


class HeuristicSearch:
    """
    对 Carp 问题进行基本启发式搜索，尝试得到一个基本的解。
    启发式不是元启发式，启发式是根据领域知识人类编撰的，但是无法被证明最优性，甚至不知道是不是朝着最优方向走。
    下面的 LocalSearch 类和 EvolutionarySearch 才是元启发式算法。
    Carp 的启发式算法主要有 augment-merge, path-scanning, construct and strike, Ulusoy 's tour splitting, augment-insert 等。
    我们重点掌握实现了 path-scanning.
    由于 carp 问题是 中国乡土(rural，表示并不是所有地方都是邮递员管辖的区域)邮递员问题 的一般形式，所以也可以用 path-scanning 解 中国乡土邮递员问题， 方法是设置 capacity 为无穷大。
    如果需要解更加特殊的 中国邮递员问题 ， 可以 每一条边都设置一个任务。
    """

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


class LocalSearch:
    """
    使用局部搜索
    """
    pass


class EvolutionarySearch:
    """
    使用演化计算
    """

    def __init__(self, carp_instance, population_size, budget, probability_local_search):
        self.carp_instance = carp_instance
        self.population_size = population_size
        self.budget = budget
        self.probability_local_search = probability_local_search

    def optimize(self):
        pass

def main():
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
    # carp_instance2 = carp_instance.copy()
    # carp_instance2.capacity = 3
    # carp_solver = HeuristicSearch(carp_instance2)
    carp_solver = HeuristicSearch(carp_instance)
    solution = carp_solver.path_scanning()
    assert solution.costs == carp_instance.costs_of(solution.routes)
    print(solution)
if __name__ == '__main__':
    main()
