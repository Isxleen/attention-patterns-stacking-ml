import pandas as pd
import numpy as np
import os

class DataIngestor:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.supported_tasks = {
            "flanker": "Raw_Flanker.xlsx",
            "stroop": "Raw_Stroop.xlsx",
            "sart": "Raw_Sart.xlsx"
        }

    def load_task(self, task_name):
        if task_name not in self.supported_tasks:
            raise ValueError(f"Tarea no soportada: {task_name}")
        
        file_path = os.path.join(
            self.raw_data_path,
            self.supported_tasks[task_name]
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el archivo {file_path}")
        
        df = pd.read_excel(file_path)
        df["task"] = task_name

        return df
    
    def load_all_tasks(self):
        data = {}
        for task in self.supported_tasks:
            data[task] = self.load_task(task)
        
        return data

    def load_and_merge(self):
        dfs = []
        for task in self.supported_tasks:
            df = self.load_task(task)
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)


