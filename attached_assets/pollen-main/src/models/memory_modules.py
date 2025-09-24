import json

class EpisodicMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.logs = []

    def add(self, experience):
        if len(self.logs) >= self.capacity:
            self.logs.pop(0)
        self.logs.append(experience)

    def recall(self):
        return self.logs

class LongTermMemory:
    def __init__(self, path):
        self.path = path
        self.memory = self.load_memory()

    def load_memory(self):
        try:
            with open(self.path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def update(self, key, value):
        self.memory[key] = value
        self.save_memory()

    def recall(self, key):
        return self.memory.get(key, None)

    def save_memory(self):
        with open(self.path, 'w') as f:
            json.dump(self.memory, f)

class ContextualMemory:
    def __init__(self):
        self.memory = {}

    def add(self, embedding, text):
        self.memory[tuple(embedding)] = text

    def retrieve(self, embedding):
        return self.memory.get(tuple(embedding), None)