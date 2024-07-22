import json
import os
import numpy as np

class LogPreprocessingDetails:
    def __init__(self, json_path, id, condition, week):
        self.json_path = json_path
        self.id = str(id)
        self.condition = condition
        self.week = week
        self.logs = self.load_preprocessing_details()

    def load_preprocessing_details(self):
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                return json.load(f)
        else:
            return {}

    def save_preprocessing_details(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.logs, f, indent=4)

    def initialize_log_structure(self):
        if self.id not in self.logs:
            self.logs[self.id] = {}
        if self.condition not in self.logs[self.id]:
            self.logs[self.id][self.condition] = {}
        if self.week not in self.logs[self.id][self.condition]:
            self.logs[self.id][self.condition][self.week] = {}

    def log_detail(self, key, value):
        self.initialize_log_structure()
        if isinstance(value, np.ndarray):
            value = value.tolist()  # Convert numpy arrays to lists
        self.logs[self.id][self.condition][self.week][key] = value

    def get_log(self):
        self.initialize_log_structure()
        return self.logs[self.id][self.condition][self.week]