import fire
import os
import argparse

from core.main import high_quality_dataset_construction, reliable_data_driven_opt


def run_hdc():
    # TODO @yongfeng
    pass


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
