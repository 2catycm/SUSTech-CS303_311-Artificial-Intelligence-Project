import unittest
from pathlib import Path

from src.project3_容量限制约束弧路径问题.submit.CARP_solver import CarpInstance, HeuristicSearch, SolutionOperators

data_location = "CARP_samples"
cwd = Path('.')
cwd = cwd.absolute()
cwd /= data_location
dats = list(cwd.rglob("*.dat"))
solution_operators = SolutionOperators()


class MyTestCase(unittest.TestCase):
    def test_can_path_scanning(self):
        for i, dat in enumerate(dats):
            carp_instance = CarpInstance().with_file(str(dat)).with_distances_calculated()
            carp_solver = HeuristicSearch(carp_instance)
            solution = carp_solver.path_scanning()
            self.assertEqual(carp_instance.costs_of(solution.routes), solution.costs)

    def test_can_ulusoy_split(self):
        for i, dat in enumerate(dats):
            carp_instance = CarpInstance().with_file(str(dat)).with_distances_calculated()
            carp_solver = HeuristicSearch(carp_instance)
            solution = carp_solver.path_scanning()
            task_edges = solution_operators.merge(solution.routes)
            routes, costs = solution_operators.ulusoy_split(task_edges, carp_instance)
            self.assertEqual(carp_instance.costs_of(routes), costs)


if __name__ == '__main__':
    unittest.main()
