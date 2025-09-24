import unittest
from flask import Flask
from src.api.main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_propose_task(self):
        response = self.app.post('/propose-task', json={'input_text': 'Generate a task for the model.'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('task', response.json)

    def test_solve_task(self):
        response = self.app.post('/solve-task', json={'input_text': 'Solve this math problem: 2+2'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('solution', response.json)

if __name__ == '__main__':
    unittest.main()