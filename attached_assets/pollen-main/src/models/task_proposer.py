import torch
import torch.nn as nn

class TaskProposer(nn.Module):
    def __init__(self, model_name="pollen-adaptive-intelligence"):
        super(TaskProposer, self).__init__()
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', model_name)
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model_name)

    def propose_task(self, context):
        inputs = self.tokenizer(context, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        task = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return task