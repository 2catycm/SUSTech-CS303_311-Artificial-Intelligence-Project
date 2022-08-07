import random
import math
import numpy as np
import re
import copy
from typing import List

# 日志输出模块：用于debug
# 参考 https://blog.m-jay.cn/?p=410
import logging

# 此处修改颜色
FMTDCIT = {
    'ERROR': "\033[31mERROR\033[0m",
    'INFO': "\033[37mINFO\033[0m",
    'DEBUG': "\033[1mDEBUG\033[0m",
    'WARN': "\033[33mWARN\033[0m",
    'WARNING': "\033[33mWARNING\033[0m",
    'CRITICAL': "\033[35mCRITICAL\033[0m",
}


class Filter(logging.Filter):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def filter(self, record: logging.LogRecord) -> bool:
        record.levelname = FMTDCIT.get(record.levelname)
        return True


filter = Filter()


def getLogger(
        name: str,
        level: int = logging.INFO,
        fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        fmt_date: str = "%H:%M:%S"
) -> logging.Logger:
    formatter = logging.Formatter(fmt, fmt_date)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    ch.addFilter(filter)

    ret = logging.getLogger(name)
    ret.setLevel(level)
    ret.addHandler(ch)
    return ret


# logger = getLogger(__name__, logging.DEBUG)  # 调试使用这个级别
logger = getLogger(__name__, logging.CRITICAL) # 上交代码时使用这个级别

# 时间控制模块
import time


class TimeController():
    def __init__(self):
        self.time_start = 0.0
        self.time_limit = 0.0

    def start_to_time(self):
        self.time_start: float = time.time()

    def set_time_limit(self, time_limit):
        self.time_limit = time_limit

    def have_more_time(self, ratio=0.9, 漏统计的时间=0.1):
        used = time.time() - self.time_start
        return used + 漏统计的时间 < ratio * self.time_limit

    def get_time_used(self):
        return time.time() - self.time_start


time_controller = TimeController()


# 基础图论算法
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


# 基础数据结构
class LruCache:
    pass


# 问题定义
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


# 问题的解的定义
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


# 问题的解的变换算子
class SolutionOperators:
    """
    对 Carp 解进行修改，以便优化算法可以优化。
    """

    # 重点：解的排序分割表示（routes， 任务边集合的集合）和排序表示（route或者trip，或者giant route， 是任务边的集合） 是两种等效的编码。
    # 但是转换的复杂度不同。
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
            return self._ulusoy_split_SSP(task_edges, carp_instance)
            # return self._ulusoy_split_DP(task_edges, carp_instance)
        else:
            # return self._ulusoy_split_DP(task_edges, carp_instance)
            return self._ulusoy_split_SSP(task_edges, carp_instance)

    def _ulusoy_split_SSP(self, task_edges, carp_instance):
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
            task_edge_i = task_edges[i]
            current_end = task_edge_i[1]  # task_edge_i.end
            cost = distances[depot, task_edge_i[0]]  # task_edge_i.start
            cost += task_edge_i[2]  # task_edge_i.cost
            cost += distances[current_end, depot]
            demand = task_edge_i[3]  # task_edge_i.demand
            assert demand <= capacity
            digraph[2 * i + 1].append([2 * i + 2, cost])  # 加入这一种到达方案
            # 初始代价
            for j in range(i + 1, N):
                task_edge_j = task_edges[j]
                # 两两组合的一种枚举。包括自己到自己。
                # 表示从这个任务开始，到这个任务结束，中间没有回仓库休息。
                cost -= distances[current_end, depot]
                cost += distances[current_end, task_edge_j[0]]  # task_edge_j.start
                cost += task_edge_j[2]  # task_edge_j.cost
                current_end = task_edge_j[1]  # task_edge_j.end
                cost += distances[current_end, depot]
                demand += task_edge_j[3]  # task_edge_i.demand
                if demand <= capacity:
                    # 是一种可能情况，所以加入到图中。
                    digraph[2 * i + 1].append([2 * j + 2, cost])  # 加入这一种到达方案
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
        split_points = [current + 1]
        while current > 1:
            parent = paths[current]
            assert parent != -1 and parent < current and parent % 2 == 1  # 不是没有最短路径、在前面、是奇数节点而不是偶数节点。
            split_points.insert(0, parent)
            current = parent - 1  # 上一步, 最后一步变成0
        # 4. 根据切割面获得结果。比如，上一步求出了 1, 5, 8为分割点
        costs = distances[2 * N]
        routes = []
        current = 0  # 指向 task_edges
        for i in range(1, len(split_points)):  # 切割点第一个总是1， 不管。
            split_point = split_points[i]
            route = []
            while (current + 1) * 2 < split_point:  # 比如 4<5
                route.append(task_edges[current])
                current += 1
            routes.append(route)

        return routes, costs

    def _ulusoy_split_DP(self, task_edges, carp_instance):
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
        distances = carp_instance.distances
        depot = carp_instance.depot
        capacity = carp_instance.capacity
        N = len(task_edges)  # 任务数量
        # 1. 求解容量约束的固定排序决策问题。
        opt = np.ones((N, capacity + 1)) * np.inf  # opt[i, j] 已知0：i的节点， 在j的容量限制下的最好表现
        go_back = np.zeros((N, capacity + 1), dtype=bool)  # go_back[i, j】 已知0：i的几点，在j容量下，上一步i-1的位置要不要回城
        for c in range(capacity + 1):
            if c >= task_edges[0][3]:  # task_edges[0].demand
                opt[0, c] = 0
        for i in range(1, N):
            task_edge_i_1 = task_edges[i - 1]
            task_edge_i = task_edges[i]
            for j in range(1, capacity + 1):
                # 选择回城。 opt的容量充满
                go_back_cost = distances[task_edge_i_1[1], depot]  # 从上一次的end回城
                go_back_cost += distances[depot, task_edge_i[0]]  # 来到这一次的起点
                go_back_cost += opt[i - 1, capacity]
                if j <= task_edge_i[3]:
                    opt[i, j] = go_back_cost
                    go_back[i, j] = True
                else:
                    # 选择不回城。opt的容量减少
                    remain_cost = distances[task_edge_i_1[1], task_edge_i[0]]  # 直接到达下一个任务点
                    remain_cost += opt[i - 1, j - task_edge_i[3]]  # task_edge_i.demand 注意溢出
                    # 两边之和大于第三边，有了容量约束之后，这个不成立
                    if remain_cost > go_back_cost:
                        opt[i, j] = go_back_cost
                        go_back[i, j] = True
                    else:
                        opt[i, j] = remain_cost
                        go_back[i, j] = False
        # 2. 恢复切割。 比如，上一步求出了 4, 3, 2, 1 应该分割，应该恢复容量等
        current_node = N - 1
        current_cap = capacity
        split_points = [N]
        while current_node > 0:
            assert current_cap >= 0
            if go_back[current_node, current_cap]:
                split_points.insert(0, current_node)
                current_cap = capacity
            else:
                current_cap -= task_edges[current_node][3]  # 剪掉demand
            current_node -= 1
        # 3. 根据切割面获得结果。比如，上一步求出了 0,1,2 中， 1前，2前均要分割
        routes = []
        current_node = 0
        for split_point in split_points:
            route = []
            while current_node < split_point:
                route.append(task_edges[current_node])
                current_node += 1
            routes.append(route)

        # 4. 规整化 cost
        costs = opt[N - 1, capacity]
        for i, task_edge in enumerate(task_edges):
            costs += task_edge[2]  # task_edge.cost 这是这个顺序下不可避免的cost
        costs += distances[depot, task_edges[0][0]]
        costs += distances[task_edges[-1][1], depot]
        return routes, costs

    # 第一类操作子，对排序表示的编码进行操作
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

    def better_flip(self, carp_instance: CarpInstance, task_edges: List[List[int]], i, j=None):
        old_routes, old_costs = self.ulusoy_split(task_edges, carp_instance)
        self.flip(task_edges, i)
        new_routes, new_costs = self.ulusoy_split(task_edges, carp_instance)
        if new_costs > old_costs:
            self.flip(task_edges, i)
            return old_routes, old_costs
        return new_routes, new_costs
    # 第二类操作子，比如 reversed。


solution_operators = SolutionOperators()


# 启发式搜索
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

    def maximize_dist(self, new_edge, old_edge, load, capacity, depot, distances):
        diff_distance = distances[new_edge[0], depot] - distances[old_edge[0], depot]
        return diff_distance > 0

    def minimize_dist(self, new_edge, old_edge, load, capacity, depot, distances):
        diff_distance = distances[new_edge[0], depot] - distances[old_edge[0], depot]
        return diff_distance < 0

    def maximize_yield(self, new_edge, old_edge, load, capacity, depot, distances):
        diff_ratio = new_edge[3] / new_edge[2] - old_edge[3] / old_edge[2]
        return diff_ratio > 0

    def minimize_yield(self, new_edge, old_edge, load, capacity, depot, distances):
        diff_ratio = new_edge[3] / new_edge[2] - old_edge[3] / old_edge[2]
        return diff_ratio < 0

    def half_full_dist(self, new_edge, old_edge, load, capacity, depot, distances):
        full_ratio = load / capacity
        assert 0 <= full_ratio <= 1
        diff_distance = distances[new_edge[0], depot] - distances[old_edge[0], depot]
        return diff_distance > 0 if full_ratio < 0.5 else diff_distance < 0

    def path_scanning(self, distances, task_edges, depot, capacity, evaluator_index: int = 4):
        unserviced = copy.deepcopy(task_edges)
        evaluators = [self.maximize_dist, self.minimize_dist, self.maximize_yield, self.minimize_yield,
                      self.half_full_dist]
        assert 0 <= evaluator_index < len(evaluators)
        evaluator = evaluators[evaluator_index]
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
                        elif new_dist == d and evaluator(new_edge, u, load, capacity, depot, distances):
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

    def path_scanning_old(self, carp_instance, evaluator_index: int = 4):
        distances = carp_instance.distances
        task_edges = carp_instance.task_edges
        depot = carp_instance.depot
        capacity = carp_instance.capacity
        return self.path_scanning(distances, task_edges, depot, capacity)


heuristic_search = HeuristicSearch()


# 局部搜索
class LocalSearch:
    """
    使用局部搜索。
    """

    def simulated_annealing(self, carp_instance: CarpInstance,
                            initial_solution: CarpSolution,
                            schedule=lambda t: 0.999 ** t,
                            halt=lambda T: T < 1e-7,
                            accept_rate=1,
                            log_interval=200):
        """
        模拟退火算法（分割排序版）。
            解的 DNA 表示为 完整的 分割排序表示，也就是 任务边集合的集合。
        :param carp_instance:
        :param initial_solution: 是 任务的集合。
        :param schedule:
        :param halt:
        :param accept_rate:
        :param log_interval:
        :return routes:
        :return costs:
        """
        routes, costs = initial_solution.routes, initial_solution.costs
        new_routes, new_costs = None, None
        t = 0  # time step
        # T = schedule(t)  # temperature
        better_t = lambda t: time_controller.get_time_used() / time_controller.time_limit * math.log(
            1e-7) / math.log(0.999)
        T = schedule(better_t(t))  # temperature
        while not halt(T) and time_controller.have_more_time():
            T = schedule(better_t(t))
            diff = np.inf
            trys = 10
            while diff >= 0 and trys > 0 and math.exp(-diff / T) <= random.random():
                new_routes, new_costs = self.one_step_local_search(carp_instance, solution_operators.merge(routes))
                diff = new_costs - costs
                trys -= 1
            if diff >= 0 and trys <= 0:
                continue
            routes, costs = new_routes, new_costs

            # update time and temperature
            if t % log_interval == 0:
                logger.info(f"step {t}: T={T}, current_value={costs}")
            t += 1
            T = schedule(t)
        logger.warning(f"finally: step {t}: T={T}, current_value={costs}")
        return routes, costs

    def one_step_local_search(self, carp_instance: CarpInstance, initial_task_edges: List[List[int]]):
        """
        Liu&Ray 局部搜索方法 的 第一步。同样是我们模拟退火找邻居的一步。
        :return routes:
        :return costs:
        """
        state11, i11, j11 = solution_operators.operator_interface('single_insertion', initial_task_edges)
        state12, i12, j12 = solution_operators.operator_interface('double_insertion', initial_task_edges)
        state13, i13, j13 = solution_operators.operator_interface('swap', initial_task_edges)
        routes11, costs11 = solution_operators.better_flip(carp_instance, state11, j11)
        routes12, costs12 = solution_operators.better_flip(carp_instance, state12, j12)
        routes13, costs13 = solution_operators.better_flip(carp_instance, state13, j13)
        if costs11 <= costs12 and costs11 <= costs13:
            routes, costs = routes11, costs11
        elif costs12 <= costs11 and costs12 <= costs13:
            routes, costs = routes12, costs12
        elif costs13 <= costs11 and costs13 <= costs12:
            routes, costs = routes13, costs13
        else:
            assert False
        return routes, costs

    def liu_ray_local_search(self, carp_instance: CarpInstance, initial: List[List[int]] = None):
        """
        使用 Liu&Ray 的局部搜索方法
        :return routes:
        :return costs:
        """
        routes, costs = self.one_step_local_search(carp_instance, initial)
        N_trips = len(routes)
        l_times = int(min(N_trips * (N_trips - 1) / 2, 50))  # number of attempts
        times = 0
        # 随便选择两个 route（trip）， 重新分配
        i_range = list(range(N_trips - 1))
        random.shuffle(i_range)
        try:
            for i in i_range:
                j_range = list(range(i + 1, N_trips))
                random.shuffle(j_range)
                for j in j_range:
                    times += 1
                    if times > l_times:
                        raise Exception()  # TODO

        finally:
            pass  # TODO


local_search = LocalSearch()


# 演化计算
class EvolutionarySearch:
    """
    使用演化计算
    """

    def simple_evolution_search(self, carp_instance, population_size, budget, probability_local_search):
        pass

    def liu_ray_global_search(self, carp_instance, population_size, budget, probability_local_search):
        pass

    def mei_tang_yao_global_search(self):
        pass


# 参数解析，主方法
def main():
    import argparse
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
                             'seed controls all the stochastic behaviors of your solver, such that the same random '
                             'seeds will make your solver produce the same results. If your solver is deterministic, '
                             'it still needs to accept −s <random_seed>, but can just ignore them while solving CARPs')
    parser.add_argument('--version', action='version', version='version 0.1.0')
    # 解析参数步骤
    args = parser.parse_args()
    random.seed(args.random_seed)  # 也可以不加，避免被攻击。
    time_controller.set_time_limit(args.termination)
    time_controller.start_to_time()
    carp_instance = CarpInstance().with_file(args.carp_instance).with_distances_calculated()

    solution = heuristic_search.path_scanning_old(carp_instance)
    assert solution.costs == carp_instance.costs_of(solution.routes)
    task_edges = solution_operators.merge(solution.routes)
    routes, costs = solution_operators.ulusoy_split(task_edges, carp_instance)
    initial = CarpSolution(routes, costs)
    routes, costs = local_search.simulated_annealing(carp_instance, initial)
    print(CarpSolution(routes, costs))
    logger.info("finished")


if __name__ == '__main__':
    main()
