import os
import pickle
from collections import defaultdict
from flaml import AutoML
import numpy as np
from collections import defaultdict

from utils.utils import generate_features


class AutoPipeline(object):
    def __init__(self, logger):
        self.logger = logger
        self.model = []
        
    def train(self, X_train, Y_train_df, time_budget, log_dir, model_dir):
        statistics = defaultdict(list)
        models = []
        for key in ["distr", "mono", "ret"]:
            Y_train = Y_train_df[key].values
            
            automl = AutoML()
            settings = {
                "time_budget": time_budget,
                "metric": 'mse',
                "task": 'regression',
                "estimator_list": ["lgbm"],
                "log_file_name": os.path.join(log_dir, key + ".log"),
            }
            automl.fit(X_train=X_train, y_train=Y_train,
                    **settings)
            models.append(automl)
            
            with open(os.path.join(model_dir, key + ".pkl"), "wb") as f:
                pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
        self.model = models
        
        return statistics
    
    def infer(self, xs, par_pop, par_fit):
        par_xs_features = np.array([generate_features(x) for x in par_pop]) / 100 #normalize
        xs_features = np.array([generate_features(x) for x in xs]) / 100 #normalize
        Ys_pred = np.zeros(xs_features.shape[0]) # size: [pop_size]
        Ys_pred_subs = np.zeros([xs_features.shape[0], 3]) # size: [pop_size]
        
        for i, k in enumerate(["distr", "mono", "ret"]):
            bag_models = self.model[i]
            par_sub_Ys_pred = np.zeros([len(bag_models), par_xs_features.shape[0]]) # size: [num_models, pop_size]
            for j, m in enumerate(bag_models):
                par_sub_Ys_pred[j] = m.predict(par_xs_features)
            par_sub_Ys_pred = np.clip(par_sub_Ys_pred, a_min=0,  a_max=None)
            par_sub_Ys_err = np.abs(par_sub_Ys_pred - par_fit[:,i]) # size: [num_models, pop_size]
            
            rel_score = np.zeros(len(bag_models)) # size: [num_models]
            for j in range(len(bag_models)):
                gap_mat = np.abs(par_sub_Ys_pred[j] - par_sub_Ys_pred[j].reshape(par_sub_Ys_pred[j].shape[0], 1))
                err_mat = par_sub_Ys_err[j] + par_sub_Ys_err[j].reshape(par_sub_Ys_err[j].shape[0], 1)
                rel_score[j] = np.sum(((gap_mat >= err_mat) * (gap_mat > 1e-4)) | ((gap_mat <= 1e-4) * (err_mat <= 1e-4)))
            if np.sum(rel_score) == 0:
                rel_score = np.ones(len(bag_models)) # size: [num_models]
            rel_score = rel_score / np.sum(rel_score)
            
            sub_Ys_pred = np.zeros([len(bag_models), xs_features.shape[0]]) # size: [num_models, pop_size]
            for j, m in enumerate(bag_models):
                sub_Ys_pred[j] = m.predict(xs_features)
            
            ensemble_sub_Ys_pred = np.sum(rel_score * sub_Ys_pred.T, axis=1) # size: [pop_size]
            Ys_pred += ensemble_sub_Ys_pred
            Ys_pred_subs[:,i] = ensemble_sub_Ys_pred
        
        return Ys_pred, Ys_pred_subs, None

    def load_model(self, load_model_dir, model_num):
        models = []
        for key  in ["distr", "mono", "ret"]:
            model_bags = []
            for i in range(model_num):
                path = os.path.join(load_model_dir, "bag_" + str(i))
                model_path = os.path.join(path, f"{key}.pkl")
                with open(model_path, 'rb') as f:
                    model_bags.append(pickle.load(f))
            models.append(model_bags)
        self.model = models
