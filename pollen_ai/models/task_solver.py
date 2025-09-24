import torch
import torch.nn as nn

class TaskSolver(nn.Module):
    def __init__(self, model_name="pollen-adaptive-intelligence"):
        super(TaskSolver, self).__init__()
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', model_name)
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model_name)

    def solve_task(self, task):
        inputs = self.tokenizer(task, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        solution = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return solution