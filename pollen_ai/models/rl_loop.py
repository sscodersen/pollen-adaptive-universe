import torch
import torch.nn as nn
import torch.optim as optim

class RLLoop:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, tasks, solutions):
        for task, solution in zip(tasks, solutions):
            inputs = self.tokenizer(task, return_tensors="pt")
            outputs = self.model(**inputs)
            loss = self.loss_fn(outputs.logits, solutions)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()