import copy as cp
import geatpy as ea
import logging
import numpy as np
import pandas as pd
import time


class SurrogateProblemOnline(ea.Problem):
    def __init__(self, zhiyong_df, zhunru_df, month_gap, segments_num, slb, sub, field_name, rank, target_distr, M,
                 constraint_list, zhiyong_months, zhunru_months, base_month):
        self.zhiyong_df = zhiyong_df
        self.zhunru_df = zhunru_df
        self.month_gap = month_gap
        self.field_name = field_name
        self.rank = rank
        self.segments_num = segments_num
        self.sub = sub
        self.slb = slb
        self.target_distr = target_distr
        self.M = M

        self.constraint_list = constraint_list
        self.zhiyong_months = zhiyong_months
        self.zhunru_months = zhunru_months
        self.base_month = base_month

        self.ml_dataset = pd.DataFrame(columns=["sols", "distr", "mono", "ret", "stab"], data=None, dtype=object)

        name = 'SegmentProblem'
        maxormins = [1] * self.M
        Dim = self.segments_num
        varTypes = [1] * Dim
        lb = [self.slb] * Dim
        ub = [self.sub] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim

        ea.Problem.__init__(self, name, self.M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        logging.info(f'Problem name: {self.name}, objectives: {self.M}')

    def compute_constraint(self, distr_data, risk_rank_data, stability_data, stability_data_3):
        total_cons = []
        segments_num = self.segments_num
        distribution_cons = []
        if "DISTRIBUTION" in self.constraint_list:
            month_num1 = distr_data.shape[1]
            if self.target_distr == 'DESC':
                for j in range(month_num1):
                    seg_users = distr_data[:, j]
                    distribution_cons.extend(
                        [1.1 - seg_users[i] / seg_users[i + 1] for i in range(segments_num - 1)])
            elif self.target_distr == 'ASC':
                for j in range(month_num1):
                    seg_users = distr_data[:, j]
                    distribution_cons.extend(
                        [1.1 - seg_users[i + 1] / seg_users[i] for i in range(segments_num - 1)])
            elif self.target_distr == 'NORMAL':
                for j in range(month_num1):
                    seg_users = distr_data[:, j]
                    max_bin_id = int(np.argmax(seg_users))
                    distribution_cons.extend([1.1 - seg_users[i + 1] / seg_users[i] for i in range(0, max_bin_id)])
                    distribution_cons.extend(
                        [1.1 - seg_users[i] / seg_users[i + 1] for i in range(max_bin_id, segments_num - 1)])
            elif self.target_distr == 'UNIFORM':
                for j in range(month_num1):
                    seg_users = distr_data[:, j]
                    distribution_cons.extend(
                        [0.5 - seg_users[i + 1] / seg_users[i] for i in range(segments_num - 1)])
                    distribution_cons.extend(
                        [seg_users[i + 1] / seg_users[i] - 1.5 for i in range(segments_num - 1)])
            total_cons.extend(distribution_cons)

        risk_rank_cons = []
        if "RISK_RANK" in self.constraint_list:
            for j in range(risk_rank_data.shape[1]):
                risk_ranks = risk_rank_data[:, j]
                risk_rank_cons.extend([1.5 - risk_ranks[i + 1] / risk_ranks[i] for i in range(segments_num - 1)])

            total_cons.extend(risk_rank_cons)

        stability_cons = []
        if "STABILITY" in self.constraint_list:
            month_num1 = distr_data.shape[1]
            month_num3 = stability_data.shape[1]
            stable_rate = [0.7] * segments_num
            stable_rate[0] = stable_rate[-1] = 0.8
            stable_rate_3 = [0.6] * segments_num
            stable_rate_3[0] = stable_rate_3[-1] = 0.7
            for j in range(month_num1):
                stability_cons.extend(
                    [stable_rate[i] - stability_data[i, j] / (distr_data[i, j] + 1e-8) for i in range(segments_num)])
            for j in range(month_num3):
                stability_cons.extend([
                    stable_rate_3[i] - stability_data_3[i, j] / (distr_data[i, j] + 1e-8) for i in range(segments_num)
                ])
            total_cons.extend(stability_cons)

        return total_cons, distribution_cons, risk_rank_cons, stability_cons

    def get_segments_bound(self, var, month):
        lb = 0
        segments = []
        for i in range(len(var)):
            slice_df = self.zhiyong_df[(self.zhiyong_df['month'] == month)
                                       & (self.zhiyong_df['box_index'] == var[i])]
            if slice_df is None or len(slice_df) == 0:
                raise Exception(f'month:{month}, box_index:{var[i]} has no user.')
            ub = slice_df.iloc[0]['max_value']
            segments.append([lb, ub])
            lb = ub

        return segments

    def get_user(self, segments, month, i):
        month_column_name = f'{self.field_name}_{month}'
        agg_df2 = self.zhunru_df[(self.zhunru_df[month_column_name] <= segments[i][1])
                                 & (self.zhunru_df[month_column_name] > segments[i][0])]
        user_ids = agg_df2['user_id'].tolist()

        return set(user_ids)

    def get_common_users(self, segments, month1, i1, month2, i2):
        usr_lst1 = self.get_user(segments, month1, i1)
        usr_lst2 = self.get_user(segments, month2, i2)
        num_users = len(set(usr_lst1) & set(usr_lst2))

        return num_users

    def get_rank_risks(self, var, month):
        lb = -1
        risk = []
        lift = []
        for i in range(len(var)):
            agg_df = self.zhiyong_df[(self.zhiyong_df['month'] == month)
                                     & (self.zhiyong_df['box_index'] <= var[i]) &
                                     ((self.zhiyong_df['box_index'] > lb))]
            bad_flag = agg_df['bad_flag'].sum()
            count = agg_df['number'].sum()
            risk.append(bad_flag / count)
            lb = var[i]

        return risk

    def get_risk_rank_data(self, var):
        risk_rank_data = []
        for j in range(0, len(self.zhiyong_months), self.month_gap):
            risks = self.get_rank_risks(var, self.zhiyong_months[j])
            risk_rank_data.append(risks)
        return np.array(risk_rank_data, dtype=object).T

    def get_distribution_data(self, segments):
        distr_data = [[] for i in range(self.segments_num)]
        for i in range(self.segments_num):
            for j in range(0, len(self.zhunru_months), self.month_gap):
                user_ij = self.get_user(segments, self.zhunru_months[j], i)
                distr_data[i].append(len(user_ij))

        return np.array(distr_data, dtype=object)

    def get_stability_data(self, segments):
        stability_data = [[] for i in range(self.segments_num)]
        stability_data_3 = [[] for i in range(self.segments_num)]
        for i in range(self.segments_num):
            for j in range(0, len(self.zhunru_months) - 1, self.month_gap):
                user_i_cur = self.get_user(segments, self.zhunru_months[j], i)
                user_i_nxt = self.get_user(segments, self.zhunru_months[j + 1], i)
                stability_data[i].append(len(user_i_nxt & user_i_cur))

            for j in range(0, len(self.zhunru_months) - 3, self.month_gap):
                user_i_cur = self.get_user(segments, self.zhunru_months[j], i)
                user_i_nxt = self.get_user(segments, self.zhunru_months[j + 3], i)
                stability_data_3[i].append(len(user_i_nxt & user_i_cur))

        return np.array(stability_data, dtype=object), np.array(stability_data_3, dtype=object)

    def evalVars(self, vars):
        ObjV = []
        evaluate_st = time.time()
        for var in vars:
            segments = self.get_segments_bound(var, self.base_month)

            distr_data = np.array([])
            if "DISTRIBUTION" in self.constraint_list:
                distr_data = self.get_distribution_data(segments)

            risk_rank_data = np.array([])
            if "RISK_RANK" in self.constraint_list:
                risk_rank_data = self.get_risk_rank_data(var)

            stability_data, stability_data_3 = np.array([]), np.array([])
            if "STABILITY" in self.constraint_list:
                stability_data, stability_data_3 = self.get_stability_data(segments)

            var_cons, distribution_cons, risk_rank_cons, stability_cons = self.compute_constraint(distr_data,
                                                                                                  risk_rank_data,
                                                                                                  stability_data,
                                                                                                  stability_data_3)
            var_obj = sum([em if em > 0 else 0 for em in var_cons])
            var_obj2 = sum([1 if em > 0 else 0 for em in var_cons])
            ObjV.append([var_obj, var_obj2])
            logging.info(
                f'var:{var}, constraints: {len(var_cons)}, vio_amt:{var_obj}， vio_num: {var_obj2}, satisfy ratio: {1 - var_obj2 / len(var_cons)}'
            )

            train_data = self.generate_trainning_data(var, distribution_cons, risk_rank_cons, stability_cons)
            self.insert_into_table(train_data)
        evaluate_ed = time.time()
        logging.info(f'var evaluation: {(evaluate_ed - evaluate_st) / vars.shape[0]}s, var num: {vars.shape[0]}')
        return np.array(ObjV), None

    def generate_trainning_data(self, var, distribution_cons, risk_rank_cons, stability_cons):
        feature = str(list(var))

        def compute_viol_amt(spec_cons):
            return sum([em if em > 0 else 0 for em in spec_cons])

        distr_amt = compute_viol_amt(distribution_cons)
        risk_amt = compute_viol_amt(risk_rank_cons)
        ret_amt = compute_viol_amt(stability_cons)
        stab_amt = distr_amt + risk_amt + ret_amt

        return {"sols": str(feature), "distr": distr_amt,
                "mono": risk_amt, "ret": ret_amt,
                "stab": stab_amt}

    def insert_into_table(self, trainning_data):
        self.ml_dataset = self.ml_dataset.append(trainning_data, ignore_index=True)