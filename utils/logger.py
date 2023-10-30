import os
import json
import time
import pandas as pd


class Logger(object):
    def __init__(self, log_dir=None, task=None, algo=None, label=None, seed=2):
        stamp = "seed" + str(seed)
        self.log_dir = os.path.join(log_dir, stamp)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.start_time = time.time()
        
        self.sol = []
        self.distr_pred = []
        self.mono_pred = []
        self.ret_pred = []
        self.stab_pred = []
        self.epoch = []
        self.time = []
    
    def log_arg(self, arg):
        with open(os.path.join(self.log_dir, "params.json"), "w") as f:
            json.dump(arg, f, indent=4)
    
    def record_round(self, xs, Ys_pred_detail, Ys_pred, epoch):
        """_summary_

        Args:
            xs (np.array; 20*8): _description_
            Ys_pred_detail (np.array; 20*3): _description_
            Ys_pred (np.array; 20): _description_
        """
        self.sol += [str(x.tolist()) for x in xs]
        self.distr_pred += Ys_pred_detail[:,0].tolist()
        self.mono_pred += Ys_pred_detail[:,1].tolist()
        self.ret_pred += Ys_pred_detail[:,2].tolist()
        self.stab_pred += Ys_pred.tolist()
        self.epoch += [epoch] * 20
        self.time += [time.time() - self.start_time] * 20
    
    def save_results(self):
        results = {
            "sols": self.sol,
            "distr_pred": self.distr_pred,
            "mono_pred": self.mono_pred,
            "ret_pred": self.ret_pred,
            "stab_pred": self.stab_pred,
            "step": self.epoch,
            "time": self.time,
        }
        result_df = pd.DataFrame(results)
        result_df.to_csv(os.path.join(self.log_dir, "results.csv"), index=False)
    
    def print_best_sol(self):
        opt_idx = self.stab_pred.index(min(self.stab_pred))
        print("------------")
        print("the optimal solution: " + str(self.sol[opt_idx]))
        print("the predicted stability regret of optimal solution: " + str(self.stab_pred[opt_idx]))