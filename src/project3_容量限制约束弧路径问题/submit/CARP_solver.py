import argparse
import time
import random

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
        self.graph = []  # 每一个是邻接表。邻居的表示是 other, cost, demand. 为了效率不做面向对象。
    def with_file(self, filename: str):
        with open(filename) as file:
            lines = file.readlines()
        self.name = lines[0].split()[2]
        self.vertices = int(lines[1].split()[2])
        self.depot = int(lines[2].split()[2])
        self.required_edges = int(lines[3].split()[3])
        self.non_required_edges = int(lines[4].split()[3])
        self.vehicles = int(lines[5].split()[2])
        self.capacity = int(lines[6].split()[2])
        self.total_cost_of_required_edges = int(lines[7].split()[6])
        # self.graph = [[]] * (self.vertices + 1) # 不能这样写，这样 地址是一样的。
        self.graph = [[] for i in range(self.vertices+1)]

        for line in lines[9:]:  # 直接开始读数据
            if line == "END\n":
                break
            elements = list(map(int, line.split()))
            self.graph[elements[0]] += [[elements[1], *elements[2:]]]
            self.graph[elements[1]] += [[elements[0], *elements[2:]]]
        return self
    def __str__(self):
        last_endl = 0
        s = "carp_instance("
        for name, value in vars(self).items():
            if name=='graph': continue
            s+=f"{name}={value}, "
            if len(s)-last_endl >=100:
                last_endl = len(s)
                s+="\n"
        s+="\ngraph=[adj[edge[other, cost, demand], edge[...]], adj[...]]\n"
        for v in range(1, self.vertices+1):
            s+=f"adj({v}): {self.graph[v]}\n"
        return s+")"

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
    carp_instance = CarpInstance().with_file(args.carp_instance)
    print(carp_instance)