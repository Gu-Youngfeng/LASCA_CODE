import numpy as np
import random
import pandas as pd

def setup_seed(seed=2):
    """setup seed for random procedure

    Args:
        seed (int, optional): Ramdom seed, set default as 2.
    """
    np.random.seed(seed)
    random.seed(seed)

def load_dataset(df_zhiyong_path, df_zhunru_path, is_local):
    df_zhiyong = pd.read_csv(df_zhiyong_path)
    df_zhunru = pd.read_csv(df_zhunru_path)
    df_zhiyong['disburse_month'] = df_zhiyong['disburse_month'].astype(str)
    
    return df_zhiyong, df_zhunru

def get_prebins_data(df_zhiyong, base_month):
    slice_df = df_zhiyong[(df_zhiyong['disburse_month'] == base_month) & (df_zhiyong['pd_seg'] >= 0)]
    sorted_slice_df = slice_df.sort_values(by='pd_seg')

    return sorted_slice_df["count"].to_list()

def load_init_x_y(csv_data_path, seg):
    """load a population x and the fitness (in terms of four part) from csv file
    turn bin_size to seg_point

    Args:
        csv_data_path (str): _description_

    Returns:
        np.array, np.array: Xs (pop_size * var_size), and Ys (pop_size * 4)
    """
    df = pd.read_csv(csv_data_path)[seg[0]:seg[1]]
    df["sols"] = df["sols"].map(lambda s: np.fromstring(s.strip("[]"), sep=','))
    Xs = np.array(df["sols"].values.tolist(), dtype=np.int32)
    # Xs = np.cumsum(Xs, axis=1) - 1
    Ys = np.array(df[["distr", "mono", "ret", "stab"]])

    return Xs, Ys

def generate_features(var):
        """ 构建特征向量 """
        feature = []
        for i in range(len(var)):
            if i == 0:
                feature.append(var[i] + 1)
            else:
                feature.append(var[i]-var[i-1])
        return feature
