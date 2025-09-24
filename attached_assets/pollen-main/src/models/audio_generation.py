import torch
import torch.nn as nn
import requests

class AudioGenerator(nn.Module):
    def __init__(self, model_name="pollen-adaptive-intelligence"):
        super(AudioGenerator, self).__init__()
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', model_name)
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model_name)

    def generate_audio(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        audio = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return audio

class MusicGenerator(nn.Module):
    def __init__(self, model_name="pollen-adaptive-intelligence"):
        super(MusicGenerator, self).__init__()
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', model_name)
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model_name)

    def generate_music(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        music = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return music

    def fetch_music_from_ace_step(self, input_text):
        url = "https://huggingface.co/spaces/ACE-Step/ACE-Step"
        response = requests.post(url, json={"input": input_text})
        return response.json()

    def integrate_with_model(self, input_text):
        generated_text = self.generate_music(input_text)
        ace_step_response = self.fetch_music_from_ace_step(input_text)
        integrated_output = self.model.generate(prompt=generated_text + ace_step_response["output"])
        return self.tokenizer.decode(integrated_output[0], skip_special_tokens=True)