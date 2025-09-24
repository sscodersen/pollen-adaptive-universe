import torch
import torch.nn as nn

class AdCreator(nn.Module):
    def __init__(self, model_name="pollen-adaptive-intelligence"):
        super(AdCreator, self).__init__()
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', model_name)
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model_name)

    def create_ad(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        ad = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return ad