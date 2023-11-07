import fire
import os
import argparse
import logging
from core.main import high_quality_dataset_construction, reliable_data_driven_opt
logging.basicConfig(level=logging.INFO)


def run_hdc(task="demo", input1='data/demo/df1_demo.csv', input2='data/demo/df2_demo.csv',
            hdc_path='data/demo/hdc.csv'):
    config = {
        "task": task,
        "input1": input1,
        "input2": input2,
        "hdc_path": hdc_path
    }

    high_quality_dataset_construction(config)


def run_rdo(task="demo", hdc_path="data/demo/hdc.csv", log_dir="log/demo/rdo", model_dir="model/demo"):
    config = {
        "task": task,
        "hdc_path": hdc_path,
        "log_dir": log_dir,
        "model_dir": model_dir,
        "base_model_num": 20
    }

    reliable_data_driven_opt(config)


if __name__ == "__main__":
    fire.Fire()
