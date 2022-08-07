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
            carp_solver = HeuristicSearch()
            solution = carp_solver.path_scanning_old()
            self.assertEqual(carp_instance.costs_of(solution.routes), solution.costs)

    def test_can_ulusoy_split(self):
        for i, dat in enumerate(dats):
            carp_instance = CarpInstance().with_file(str(dat)).with_distances_calculated()
            carp_solver = HeuristicSearch()
            solution = carp_solver.path_scanning_old(carp_instance)
            task_edges = solution_operators.merge(solution.routes)
            routes, costs = solution_operators.ulusoy_split(task_edges, carp_instance)
            self.assertEqual(carp_instance.costs_of(routes), costs)

    def test_can_operate(self):
        for i, dat in enumerate(dats):
            carp_instance = CarpInstance().with_file(str(dat)).with_distances_calculated()
            carp_solver = HeuristicSearch()
            solution = carp_solver.path_scanning_old()
            task_edges_old = solution_operators.merge(solution.routes)
            task_edges_new, i, j = solution_operators.operator_interface('single_insertion', task_edges_old)
            if i!=j:
                self.assertNotEqual(task_edges_new, task_edges_old)

            task_edges_old = task_edges_new
            task_edges_new, i, j = solution_operators.operator_interface('double_insertion', task_edges_old)
            if i!=j:
                self.assertNotEqual(task_edges_new, task_edges_old)

            task_edges_old = task_edges_new
            task_edges_new, i, j = solution_operators.operator_interface('swap', task_edges_old)
            if i != j:
                self.assertNotEqual(task_edges_new, task_edges_old)

            task_edges_old = task_edges_new
            task_edges_new, i, j = solution_operators.operator_interface('flip', task_edges_old) # 这一步要求deep copy
            if i != j:
                self.assertNotEqual(task_edges_new, task_edges_old)


if __name__ == '__main__':
    unittest.main()
