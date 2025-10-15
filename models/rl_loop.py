"""
Reinforcement Learning Loop for Pollen AI
Implements training and optimization mechanisms
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Any, Optional


class RLLoop:
    """
    Reinforcement Learning Loop for continuous model improvement.
    Trains the model based on user feedback and interactions.
    """
    
    def __init__(
        self, 
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        learning_rate: float = 2e-5
    ):
        self.model = model
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        
        # Create optimizer if model is provided
        if model and not optimizer:
            self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer
        
        self.training_history = []
    
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Perform a single training step.
        Returns the loss value.
        """
        if not self.model or not self.optimizer:
            return 0.0
        
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        
        loss.backward()
        self.optimizer.step()
        
        loss_value = loss.item()
        self.training_history.append(loss_value)
        
        return loss_value
    
    def train_batch(self, tasks: List[str], solutions: List[str], tokenizer: Any = None) -> List[float]:
        """
        Train on a batch of tasks and solutions.
        Returns list of loss values.
        """
        if not self.model or not tokenizer:
            return []
        
        losses = []
        
        for task, solution in zip(tasks, solutions):
            # Tokenize inputs
            inputs = tokenizer(task, return_tensors="pt", padding=True, truncation=True)
            targets = tokenizer(solution, return_tensors="pt", padding=True, truncation=True)
            
            # Train step
            loss = self.train_step(inputs['input_ids'], targets['input_ids'])
            losses.append(loss)
        
        return losses
    
    def evaluate(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Evaluate the model without training.
        Returns the loss value.
        """
        if not self.model:
            return 0.0
        
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
        
        return loss.item()
    
    def get_training_stats(self) -> dict:
        """Get training statistics"""
        if not self.training_history:
            return {
                "total_steps": 0,
                "average_loss": 0.0,
                "min_loss": 0.0,
                "max_loss": 0.0
            }
        
        return {
            "total_steps": len(self.training_history),
            "average_loss": sum(self.training_history) / len(self.training_history),
            "min_loss": min(self.training_history),
            "max_loss": max(self.training_history),
            "recent_loss": self.training_history[-1] if self.training_history else 0.0
        }
    
    def reset_history(self) -> None:
        """Reset training history"""
        self.training_history = []
    
    def update_learning_rate(self, new_lr: float) -> None:
        """Update the learning rate"""
        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
