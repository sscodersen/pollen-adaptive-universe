import torch
import torch.nn as nn

class GameGenerator(nn.Module):
    def __init__(self, model_name="pollen-adaptive-intelligence"):
        super(GameGenerator, self).__init__()
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', model_name)
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model_name)

    def generate_game_level(self, input_text):
        inputs = self.tokenizer(input