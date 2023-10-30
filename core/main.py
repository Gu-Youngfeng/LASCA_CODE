import os
import time
import numpy as np
import pandas as pd
import geatpy as ea

from core.model import AutoPipeline
from core.optimizer import EvoBinningSolver
from utils.logger import Logger
from core.task import SurrogateProblemOffline
from utils.utils import load_init_x_y, generate_features


def high_quality_dataset_construction(args):
    # TODO @yongfeng
    # user_data = args.user_data_path
    # pass
    # hdc_data = 
    # hdc_data.to_csv(args.hdc_path, index=False)
    return


def train_basemodels(config):
    bags_cnt = config["base_model_num"]
    df = pd.read_csv(config["hdc_path"])
    for i in range(bags_cnt):
        df_bag = df.sample(frac=0.7)
        log_dir = os.path.join(config["log_dir"], "train", "bag_"+str(i))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        model_dir = os.path.join(config["model_dir"], "bag_"+ str(i))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        df_bag["sols"] = df_bag["sols"].map(lambda s: np.fromstring(s.strip("[]"), sep=','))
        X_train = np.array(df_bag["sols"].values.tolist(), dtype=np.float32)
        X_train = np.array([generate_features(x) for x in X_train]) / 100
        Y_train_df = df_bag.drop(columns=["sols"])

        pl = AutoPipeline(logger = None)
        stat = pl.train(X_train, Y_train_df, 50, log_dir, model_dir)


def reliable_data_driven_opt(config):
    # train the base models if not exist
    if not os.path.exists(config["model_dir"]):
        train_basemodels(config)
    
    # optimizer config
    lb = 0
    ub = 100
    segments_num = 8
    
    logger = Logger(log_dir=config["log_dir"],
                    task=config["task"],
                    algo="rdo",
                    label="demo",
                    seed=2)
    logger.log_arg(config)
    
    # Initialization    
    pl = AutoPipeline(logger=logger)
    surrogate_task = SurrogateProblemOffline(
                                    segments_num=segments_num,
                                    slb=lb,
                                    sub=ub,
                                    M=2,
                                    automl_pl=pl,
                                    logger=logger)
    pl.load_model(config["model_dir"], config["base_model_num"])
    
    optimizer = EvoBinningSolver(
        problem=surrogate_task,
        population=ea.Population(Encoding='RI', NIND=20),
        MAXGEN=50,
        slb=lb,
        sub=ub,
        mut_rate=0.1,
        xo_rate=0.4)

    best_bins, best_ys = load_init_x_y(config["hdc_path"], [0, 20])
    surrogate_task.curr_pop = best_bins
    surrogate_task.curr_fitness = best_ys

    logger.start_time = time.time()
    res = ea.optimize(optimizer, prophet=best_bins, saveFlag=False, drawLog=False, drawing=0, verbose=False)
    logger.save_results()
    logger.print_best_sol()


if __name__ == "__main__":
    pass
