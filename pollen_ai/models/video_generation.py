import torch
import torch.nn as nn

class VideoGenerator(nn.Module):
    def __init__(self, model_name="pollen-adaptive-intelligence"):
        super(VideoGenerator, self).__init__()
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', model_name)
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model_name)

    def generate_video(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        video = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return video