import geatpy as ea
import os
import logging
import numpy as np
import pandas as pd


class SurrogateProblemOffline(ea.Problem):
    def __init__(self,
                 segments_num,
                 slb,
                 sub,
                 M,
                 automl_pl,
                 logger):
        self.segments_num = segments_num
        self.sub = sub
        self.slb = slb
        self.M = M

        self.automl_pl = automl_pl
        self.logger = logger
        
        self.eval_epoch = 0
        self.curr_pop = []
        self.curr_fitness = []

        name = 'SurrogateSegmentProblem'
        maxormins = [1] * self.M
        Dim = self.segments_num
        varTypes = [1] * Dim
        lb = [self.slb] * Dim
        ub = [self.sub] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        
        ea.Problem.__init__(self, name, self.M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        logging.info(f'Problem name: {self.name}, objectives: {self.M}')
        
    def evalVars(self, vars):
        infer_obj, Ys_pred_subs, _ = self.automl_pl.infer(vars, self.curr_pop, self.curr_fitness)
        self.curr_pop = vars
        self.curr_fitness = Ys_pred_subs
        self.logger.record_round(vars, Ys_pred_subs, infer_obj, self.eval_epoch)
        
        ObjV_pred = np.expand_dims(infer_obj, axis=1)
        ObjV_pred = np.concatenate((ObjV_pred, ObjV_pred), axis=1)
        
        self.eval_epoch += 1
        
        return ObjV_pred, None
