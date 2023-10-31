import os
import time
import logging
import numpy as np
import pandas as pd
import geatpy as ea

from core.model import AutoPipeline
from core.optimizer import EvoBinningOptimizer, PDAInitializer
from utils.logger import Logger
from core.task import SurrogateProblemOffline
from core.task_hdc import SurrogateProblemOnline
from utils.utils import load_init_x_y, generate_features


def load_dataset(df1_path, df2_path):
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    df1['month'] = df1['month'].astype(str)
    logging.info(f'[LOCAL TABLE]: df1: {df1_path}, {len(df1)}, zhunru_table: {df2_path}, {len(df2)}')
    logging.info(f'[LOCAL TABLE]: df2: {df1.dtypes}, zhunru dtypes: {df2.dtypes}')
    return df1, df2


def get_prebins_data(df1, base_month):
    slice_df = df1[(df1['month'] == base_month) & (df1['box_index'] >= 0)]
    sorted_slice_df = slice_df.sort_values(by='box_index')
    logging.info(f'slice_df: {sorted_slice_df["box_index"].to_list()}')

    return sorted_slice_df["number"].to_list()


def high_quality_dataset_construction(config):
    # load datasets
    df1, df2 = load_dataset(config['input1'], config['input2'])
    fied_name = "demo"
    constraint_list = ["DISTRIBUTION", "RISK_RANK", "STABILITY"]
    segments_num = 8
    lb = 0
    ub = 99
    base_month = "197010"
    month_gap = 4
    target_distr = "NORMAL"
    rank_order = True
    zhiyong_months = ["197010", "197011", "197012"]
    zhunru_months = ["197101", "197102", "197103", "197104", "197105", "197106", "197107", "197108"]
    init_strategy = "NORMAL"
    pop_size = 20

    # Initialization
    evobinning_task = SurrogateProblemOnline(
        zhiyong_df=df1,
        zhunru_df=df2,
        month_gap=month_gap,
        segments_num=segments_num,
        slb=lb,
        sub=ub,
        field_name=fied_name,
        rank=rank_order,
        target_distr=target_distr,
        M=2,
        constraint_list=constraint_list,
        zhiyong_months=zhiyong_months,
        zhunru_months=zhunru_months,
        base_month=base_month,
    )

    optimizer = EvoBinningOptimizer(
        problem=evobinning_task,
        population=ea.Population(Encoding='RI', NIND=pop_size),
        MAXGEN=50,
        slb=lb,
        sub=ub,
        mut_rate=0.1,
        xo_rate=0.4)

    prebins_data = get_prebins_data(df1, base_month)
    logging.info(f'prebins_data: {prebins_data}')
    generator = PDAInitializer(prebins_data=prebins_data,
                               init_strategy=init_strategy,
                               segments_num=segments_num,
                               pop_size=pop_size)
    init_population = generator.generate(disturb_ratio=0.1)

    res = ea.optimize(optimizer, prophet=init_population, saveFlag=False, drawLog=False, drawing=0, verbose=False)
    logging.info(f'collected binning data:\n{evobinning_task.ml_dataset}')
    evobinning_task.ml_dataset.to_csv(config['hdc_path'], index=False)


def train_basemodels(config):
    bags_cnt = config["base_model_num"]
    df = pd.read_csv(config["hdc_path"])
    for i in range(bags_cnt):
        df_bag = df.sample(frac=0.7)
        log_dir = os.path.join(config["log_dir"], "train", "bag_" + str(i))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        model_dir = os.path.join(config["model_dir"], "bag_" + str(i))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        df_bag["sols"] = df_bag["sols"].map(lambda s: np.fromstring(s.strip("[]"), sep=','))
        X_train = np.array(df_bag["sols"].values.tolist(), dtype=np.float32)
        X_train = np.array([generate_features(x) for x in X_train]) / 100
        Y_train_df = df_bag.drop(columns=["sols"])

        pl = AutoPipeline(logger=None)
        stat = pl.train(X_train, Y_train_df, 50, log_dir, model_dir)


def reliable_data_driven_opt(config):
    # trian the base models if not exist
    if not os.path.exists(config["model_dir"]):
        train_basemodels(config)

    # optimizer config
    lb = 0
    ub = 99
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

    optimizer = EvoBinningOptimizer(
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
    res = ea.optimize(optimizer, prophet=best_bins, saveFlag=False, drawLog=False, drawing=0,
                      verbose=False)
    logger.save_results()
    logger.print_best_sol()


if __name__ == "__main__":
    pass
