import argparse
import time
import random
import math
import numpy as np
import re
import copy
from typing import List


def floyd(N, G):
    """
    解决多源最短路问题的弗洛伊德算法。
    弗洛伊德算法正确性证明：
        首先，假设图无负环，否则存在无穷小的最短路。
        则任何节点到任何节点，必定存在一条使用节点数小于n的最短路。
        不妨设 i 到 j 节点的最短路需要 x_ij 个节点。
        # 对于任何 (i, j), 如果 x_ij =1, 则自然就是最短路。（不需要这一句，对于有负边和环的图不成立）
        对于任意(i, j), 考虑1-n的所有节点作为中介节点参与到(i, j)的最短路，不妨设最短路中编号最大的节点为 k_ij
        只要证明 k_ij 出现之前，最外层循环一定已经在左边和右边得到了最短路。
        可以递归证明。
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

    def merge(self, routes):
        """
        从多个 route 的集合 合并(merge) 为一条 "giant route" （不符合容量约束）
        需要通过 @task_edges2routes() 方法重新获得合法的 routes 集合。
        :param routes:
        :return task_edges: 一个 giant route，是 carp 问题所有任务边的一个排列。
        """
        task_edges = [task_edge for route in routes for task_edge in route]
        return task_edges

    def ulusoy_split(self, task_edges, carp_instance):
        """
        使用 Ulusoy Split 将 “giant route" 分割为 route 的集合，使得分割后符合容量约束且 costs 最低。
        :return routes: 最优路径分割方案
        :return costs: 最优路径分割方案的代价
        """
        if carp_instance.capacity <= 2 * carp_instance.vertices:
            return self.ulusoy_split_DP(task_edges, carp_instance)
        else:
            return self.ulusoy_split_SSP(task_edges, carp_instance)

    def ulusoy_split_SSP(self, task_edges, carp_instance):
        """
        转换为图的单源最短路问题求解。
        转换为图需要 O()
        :param task_edges:
        :param carp_instance:
        :return:
        """
        distances = carp_instance.distances
        depot = carp_instance.depot
        capacity = carp_instance.capacity
        N = len(task_edges)  # 任务数量
        # 1. 建图
        digraph = [[] for i in range(2 * N + 1)]  # 节点数量比较多。 有向图。
        for i in range(N):
            task_i = task_edges[i]
            for j in range(i, N):
                task_j = task_edges[j]
                # 两两组合的一种枚举。包括自己到自己。
                # 表示从这个任务开始，到这个任务结束，中间没有回仓库休息。
                current = depot
                cost = 0
                for k in range(i, j + 1):
                    task_edge_k = task_edges[k]
                    cost += distances[current, task_edge_k[0]]  # task_edge_k.start
                    cost += task_edge_k[2]  # task_edge_k.cost
                    current = task_edge_k[1]  # task_edge_k.end
                cost += distances[current, depot]
                if cost <= capacity:
                    # 是一种可能情况，所以加入到图中。
                    digraph[task_i[0]].append([task_j[0], cost])  # 加入这一种到达方案
        for i in range(1, N):  # 1:N-1 插入0
            digraph[2 * i].append([2 * i + 1, 0])
        # 2. 求解最短路
        paths = [-1 for _ in range(2 * N + 1)]
        distances = [math.inf for _ in range(2 * N + 1)]
        distances[1] = 0
        for i in range(1, 2 * N + 1):  # 由于所有边都往后面指，所以遍历到的时候一定是最短了，就是说已经被所有可能缩短距离的边松弛过。
            for relative in digraph[i]:
                new_cost = distances[i] + relative[1]  # relative.cost
                if new_cost < distances[relative[0]]:
                    distances[relative[0]] = new_cost
                    paths[relative[0]] = i
        # 3. 恢复切割。 比如，上一步求出了 1-》4-》5-》8 的最短路
        current = 2 * N  # 比如 8
        split_points = [current]
        while current > 1:
            parent = paths[current]
            assert parent != -1 and parent < current and parent % 2 == 1  # 不是没有最短路径、在前面、是奇数节点而不是偶数节点。
            split_points.insert(0, parent)
            current = parent - 1  # 上一步, 最后一步变成0
        # 4. 根据切割面获得结果。比如，上一步求出了 1, 5, 8为分割点
        routes = []
        current = 0  # 指向 task_edges
        for i in range(1, len(split_points)):  # 切割点第一个总是1， 不管。
            split_point = split_points[i]
            route = []
            while (current + 1) * 2 < split_point:  # 比如 4<5
                route.append(task_edges[current])
                current += 1
            routes.append(route)

        return routes, distances[2 * N]

    def ulusoy_split_DP(self, task_edges, carp_instance):
        """
        本算法实现为 O(nC), 如果C不大可以考虑。
        我们使用动态规划算法实现。
        设任务集合为 T1, T2, T3, ..., Tn
        中间可以分割的间隔是 S1, S2, ..., Sn-1。每个 S 赋值0表示不分割，1表示分割
        OPT(i) 为考虑了第1:i个任务后，最优的 cost。
        如果 Si-1 决策为分割，那么OPT(i) = OPT(i-1) + dist(depot, Ti.s)+cost(Ti.s, Ti.e)+dist(Ti.e, depot)
        如果 Si-1 决策为不分割， 那么 OPT(i) = OPT(i-1) - dist(Ti-1.e, depot) + dist(Ti-1.e, Ti.s)
        无后效性的证明：
            TODO
        :param task_edges:
        :param carp_instance:
        :return:
        """
        pass

    def operator_interface(self, operation_type: str, task_edges: List[List[int]], i=-1, j=-1):
        """
        算子的非原地操作的接口。
        如果要用原地操作加快速度，直接调用下面的函数。
        :param operation_type:
        :param task_edges: 自动复制传参，不会对外面造成伤害。
        :param i:
        :param j:
        :return task_edges: 修改后的新对象。
        :return i, j: 如果没有传参数进来，返回内部随机出来的参数。
            有助于外面进行修改和重新评估，比如把方向换一下, 看看哪个更好。
        """
        task_edges = copy.deepcopy(task_edges)  # 为了避免风险，直接深拷贝
        N = len(task_edges)
        upperbound = N if operation_type != 'double_insertion' else N - 1
        if not (0 <= i < upperbound):
            i = random.randrange(0, upperbound)  # range就不用减一。
        if not (0 <= j < upperbound):
            j = random.randrange(0, upperbound)
        table = {
            'single_insertion': self.single_insertion,
            'double_insertion': self.double_insertion,
            'swap': self.swap,
            'two_opt': self.two_opt,
            'flip': self.flip,
        }
        table[operation_type](task_edges, i, j)
        return task_edges, i, j

    def single_insertion(self, task_edges: List[List[int]], i, j):
        """
        remove the i th task and reinsert it into the j th place.
        删除第i个任务，然后重新插入到第j个位置
        """
        victim = task_edges.pop(i)
        task_edges.insert(j, victim)

    def double_insertion(self, task_edges: List[List[int]], i, j):
        """
        随便选择两个连在一起的任务，然后放到另外一个位置
        """
        victim_left = task_edges.pop(i)
        victim_right = task_edges.pop(i)
        task_edges.insert(j, victim_right)
        task_edges.insert(j, victim_left)

    def swap(self, task_edges: List[List[int]], i, j):
        """
        随便把两个任务换一下。
        """
        task_edges[i], task_edges[j] = task_edges[j], task_edges[i]

    def two_opt(self):
        raise NotImplementedError()

    def flip(self, task_edges: List[List[int]], i, j=None):
        task_edge = task_edges[i]
        task_edge[0], task_edge[1] = task_edge[1], task_edge[0]  # 交换顺序。


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

    def __init__(self, carp_instance):
        self.carp_instance = carp_instance
    # def


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
    random.seed(args.random_seed)  # 也可以不加，避免被攻击。
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
