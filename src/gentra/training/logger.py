import json
import os
from datetime import datetime


class Logger:
    def __init__(self, output_dir=None, verbose=True):
        self.verbose = verbose
        self.log_history = []
        self.experiment_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def log(self, message):
        if self.verbose:
            print(message)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log_history.append(f'[{timestamp}] {message}')

    def save_log(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for entry in self.log_history:
                f.write(f'{entry}\n')

    def save_results(self, file_path, experiment_info, split_results):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('===== Experiment Parameters =====\n')
            for key, value in experiment_info.items():
                f.write(f'{key}: {value}\n')
            f.write('\n===== Split Results =====\n')
            for result in split_results:
                f.write(f"\nSplit {result['split_idx'] + 1}")
                if result.get('test_subject') is not None:
                    f.write(f" (test subject {result['test_subject']})")
                f.write('\n')
                for key, value in result['final_metrics'].items():
                    f.write(f'  {key}: {value:.4f}\n')

    def save_json(self, file_path, payload):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, default=str)
