import unittest
from src.models.task_proposer import TaskProposer
from src.models.task_solver import TaskSolver

class TestModels(unittest.TestCase):
    def test_task_proposer(self):
        proposer = TaskProposer()
        task = proposer.propose_task("Generate a task for the model.")
        self.assertIsNotNone(task)

    def test_task_solver(self):
        solver = TaskSolver()
        solution = solver.solve_task("Solve this math problem: 2+2")
        self.assertEqual(solution, "4")

if __name__ == '__main__':
    unittest.main()