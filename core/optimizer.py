import copy as cp
import geatpy as ea
import logging
import numpy as np
from scipy.stats import norm, expon, uniform


class EvoBinningSolver(ea.MoeaAlgorithm):
    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 slb=None,
                 sub=None,
                 mut_rate=0.4,
                 xo_rate=0.1,
                 **kwargs):
        super().__init__(problem=problem, population=population, MAXGEN=MAXGEN, logTras=0)

        self.LB = slb
        self.UB = sub
        self.MUT_RATE = mut_rate
        self.XO_RATE = xo_rate

        slb = 1 if self.LB is None else self.LB
        sub = 100 if self.UB is None else self.UB
        self.mutOper = RangeMutator(Pm=self.MUT_RATE, lb=slb, ub=sub)
        self.selFunc = 'tour'
        self.ndSort = ea.ndsortESS

        self.BestIndObj = 0
        self.BestInd = None
        self.viol_amount_curve = []
        self.viol_number_curve = []
        self.training_set = []
        self.training_data = []

        self.name = 'GroupESSolver-NSGA2'
        logging.info(f'solver name: {self.name}, solver params: lower bound:{self.LB}, upper bound:{self.UB}, mutation rate:{self.MUT_RATE}, crossover rate:{self.XO_RATE}, max_generation:{self.MAXGEN}')

    def reinsertion(self, population, offspring, NUM):
        population = population + offspring
        [levels, _] = self.ndSort(population.ObjV, NUM, None, population.CV, self.problem.maxormins)

        dis = ea.crowdis(population.ObjV, levels)

        population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')
        chooseFlag = ea.selecting('dup', population.FitnV, NUM)
        return population[chooseFlag]
    
    def run(self, prophetPop=None):
        population = self.population
        NIND = population.sizes
        self.initialization()
        population.initChrom()
        round_id = 0
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]
        self.call_aimFunc(population)
        [levels, _] = self.ndSort(population.ObjV, NIND, None, population.CV,
                                  self.problem.maxormins)
        population.FitnV = (1 / levels).reshape(-1, 1)
        logging.info(f'>>> Round 0, population size: {len(population.Chrom)}, mean amount: {np.mean(population.ObjV[:,0])}, mean number: {np.mean(population.ObjV[:,1])}\n')

        samples = np.concatenate((population.Chrom, population.ObjV), axis=-1)
        for ins in samples:
            self.training_set.append(ins.tolist())

        while not self.terminated(population):
            round_id += 1
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            offspring.Chrom = self.mutOper.do(offspring.Chrom)
            self.call_aimFunc(offspring)
            population = self.reinsertion(population, offspring, NIND)
            samples = np.concatenate((population.Chrom, population.ObjV), axis=-1)
            for ins in samples:
                self.training_set.append(ins.tolist())
            logging.info(f'>>> Round {round_id}, population size: {len(population.Chrom)}, mean amount: {np.mean(population.ObjV[:,0])}, mean number: {np.mean(population.ObjV[:,1])}\n')
        return self.finishing(population)


class ProphetGenerator(object):
    def __init__(self, prebins_data, init_strategy, segments_num, pop_size):
        self.data = prebins_data
        self.init_strategy = init_strategy
        self.segments_num = segments_num
        self.pop_size = pop_size
        self.repeat_num = 50

        self.prophet_chrom = None

    def generate(self, disturb_ratio, chroms=None):
        if self.init_strategy == 'ASC':
            self.prophet_chrom = self.generate_by_ascending_distribution(self.data,
                                                                       self.segments_num,
                                                                       self.pop_size,
                                                                       self.repeat_num)
        elif self.init_strategy == 'DESC':
            self.prophet_chrom = self.generate_by_descending_distribution(self.data,
                                                                  self.segments_num,
                                                                  self.pop_size,
                                                                  self.repeat_num)
        elif self.init_strategy == 'NORMAL':
            self.prophet_chrom = self.generate_by_normal_distribution(self.data,
                                                                 self.segments_num,
                                                                 self.pop_size,
                                                                 self.repeat_num)
        elif self.init_strategy == 'UNIFORM':
            self.prophet_chrom = self.generate_by_uniform_distribution(self.data,
                                                                 self.segments_num,
                                                                 self.pop_size,
                                                                 self.repeat_num)
        elif self.init_strategy == 'E2':
            self.prophet_chrom = self.generate_by_expert_experience(self.segments_num,
                                                                     chroms,
                                                                     self.pop_size,
                                                                     disturb_ratio)
        else:
            raise ValueError(f'Not Supported for unknown initialization strategy "{self.init_strategy}"')

        return self.prophet_chrom

    def disturb_chrom(self, chrom, gen_num, Pm):
        pop = [chrom]
        chrom_len = len(chrom)
        slb = 0
        sub = len(self.data)
        for i in range(gen_num - 1):
            o1 = cp.copy(chrom)
            for j in range(chrom_len - 1):
                rand = np.random.rand()
                if rand <= Pm:
                    lb = slb - 1 if j == 0 else o1[j - 1]
                    ub = sub + 1 if j == chrom_len - 1 else o1[j + 1]
                    if lb + 1 < ub:
                        candidate_set = set(range(lb + 1, ub)) - {o1[j]}
                        if len(candidate_set) == 0:
                            o1[j] = lb + 1
                        else:
                            o1[j] = np.random.choice(list(candidate_set))
            pop.append(o1)
        return np.array(pop)

    def generate_by_expert_experience(self, segments_num, chroms, size, Pm) -> np.ndarray:
        logging.info('choosing the seed chrom with the Expert Experience.')
        logging.info(f'segments_num: {segments_num}')
        logging.info(f'seed chrom: {chroms}')

        if chroms is None or len(chroms) <= 0:
            raise ValueError(f'Chrom is not given in Expert Experience (E2) strategy!')

        if size > len(chroms):
            pop_appendix = self.disturb_chrom(chroms[0], size - len(chroms), Pm)
            chroms.extend(pop_appendix.tolist())
        else:
            chroms = chroms[:size]

        logging.info(f'generated chroms: {chroms}')
        return np.array(chroms)

    def get_chrom_score(self, cdf1, cdf2, dist_type):
        assert len(cdf1) == len(cdf2), f"cdf1({len(cdf1)}) and cdf2({len(cdf2)}) are in different length"
        pdf1 = self.get_pdf(cdf1)
        pdf2 = self.get_pdf(cdf2)
        # print(f'expected pdf1:{pdf1}, actual pdf2:{pdf2}')
        kl_distance = sum([pdf1[kk] * np.log(pdf1[kk] / (pdf2[kk] + 1e-8)) for kk in range(len(pdf1))])
        vio_amt = 0
        if dist_type == 'NORMAL':
            vio_amt += max(0, cdf2[0] - (cdf2[1] - cdf2[0]))  # 第 1 个箱子要比第 2 个箱子少
            vio_amt += max(0, (1 - cdf2[-1]) - (cdf2[-1] - cdf2[-2]))  # 倒数第 1 个箱子要比倒数第 2 个箱子少
        elif dist_type == 'DESC':
            vio_amt += max(0, (cdf2[1] - cdf2[0]) - cdf2[0])  # 第 1 个箱子要比第 2 个箱子多
            vio_amt += max(0, (1 - cdf2[-1]) - (cdf2[-1] - cdf2[-2]))  # 倒数第 1 个箱子要比倒数第 2 个箱子少
        elif dist_type == 'ASC':
            vio_amt += max(0, cdf2[0] - (cdf2[1] - cdf2[0]))  # 第 1 个箱子要比第 2 个箱子少
            vio_amt += max(0, (cdf2[-1] - cdf2[-2]) - (1 - cdf2[-1]))  # 倒数第 1 个箱子要比倒数第 2 个箱子多

        return kl_distance + vio_amt

    def get_pdf(self, cdf):
        pdf = []
        for i in range(len(cdf)):
            if i == 0:
                pdf.append(cdf[i] - 0)
            else:
                pdf.append(cdf[i] - cdf[i - 1])
        pdf.append(1 - cdf[-1])
        return pdf

    def generate_possible_bins(self, data, bins_num, cdf):
        split_points = []
        prebins_num = len(data)

        cum_rate = np.cumsum(data) / sum(data)
        actual_cdf = []
        st_bin = 0
        for i in range(0, bins_num - 1):
            for j in range(st_bin, prebins_num):
                if cum_rate[j] >= cdf[i]:
                    temp_rd = np.random.uniform(low=cum_rate[j - 1], high=cum_rate[j])
                    ed_bin = j - 1 if temp_rd <= cdf[i] else j
                    split_points.append(ed_bin)
                    actual_cdf.append(cum_rate[ed_bin])
                    st_bin = ed_bin + 1
                    break
        split_points.append(prebins_num - 1)
        if len(split_points) != bins_num:
            return None, None
        for i in range(1, len(split_points)):
            if split_points[i] == split_points[i - 1]:
                return None, None
        return split_points, actual_cdf

    def generate_by_uniform_distribution(self, data, segments_num, pop_size, repeat_num=50) -> np.ndarray:
        logging.info('choosing the seed chrom with the Uniform Distribution (UF).')
        logging.info(f'segments_num: {segments_num}')

        init_chroms = []
        row_cdf = [xi / segments_num for xi in range(1, segments_num + 1)]
        expected_cdf = [cdf / row_cdf[-1] for cdf in row_cdf[:segments_num - 1]]
        for k in range(repeat_num * 5):
            possible_bin, actual_cdf = self.generate_possible_bins(data, segments_num, expected_cdf)
            if possible_bin is None or actual_cdf is None:
                continue
            kl_distance = self.get_chrom_score(expected_cdf, actual_cdf, self.init_strategy)
            init_chroms.append([possible_bin, kl_distance])

        unique_chorms = []
        for chrom in init_chroms:
            if chrom not in unique_chorms:
                unique_chorms.append(chrom)
        unique_chorms.sort(key=lambda x: x[-1])
        selected_chroms = []

        if len(unique_chorms) == 0:
            logging.info(f'failed to initialize any valuable init chroms... [random generation]')
            selected_chroms = self.random_chroms(data, segments_num, pop_size)
        elif len(unique_chorms) < pop_size:
            logging.info(f'initialize {len(unique_chorms)} init chroms (expected {pop_size})...  [disturb chroms]')
            selected_chroms.extend([unique_chorms[i][0] for i in range(len(unique_chorms))])
            pop_appendix = self.disturb_chrom(unique_chorms[0][0], pop_size - len(unique_chorms), 0.3)
            selected_chroms.extend(pop_appendix.tolist())
        else:
            logging.info(f'initialize {pop_size} init chroms...  [successful]')
            selected_chroms.extend([unique_chorms[i][0] for i in range(pop_size)])

        return np.array(selected_chroms)

    def generate_by_descending_distribution(self, data, segments_num, pop_size, repeat_num=50) -> np.ndarray:
        logging.info('choosing the seed chroms with the Inverted Triangle Distribution (IT)')
        logging.info(f'segments_num: {segments_num}')

        init_chroms = []
        for lift_ratio in np.linspace(1.1, 1.5, 10):
            row_cdf = [expon(scale=1 / np.log(lift_ratio)).cdf(xi) for xi in range(1, segments_num + 1)]
            expected_cdf = [cdf / row_cdf[-1] for cdf in row_cdf[:segments_num - 1]]
            for k in range(repeat_num):
                possible_bin, actual_cdf = self.generate_possible_bins(data, segments_num, expected_cdf)
                if possible_bin is None or actual_cdf is None:
                    continue
                chrom_score = self.get_chrom_score(expected_cdf, actual_cdf, self.init_strategy)
                init_chroms.append([possible_bin, chrom_score])

        unique_chorms = []
        for chrom in init_chroms:
            if chrom not in unique_chorms:
                unique_chorms.append(chrom)
        unique_chorms.sort(key=lambda x: x[-1])
        selected_chroms = []

        if len(unique_chorms) == 0:
            logging.info(f'failed to initialize any valuable init chroms... [random generation]')
            selected_chroms = self.random_chroms(data, segments_num, pop_size)
        elif len(unique_chorms) < pop_size:
            logging.info(f'initialize {len(unique_chorms)} init chroms (expected {pop_size})...  [disturb chroms]')
            selected_chroms.extend([unique_chorms[i][0] for i in range(len(unique_chorms))])
            pop_appendix = self.disturb_chrom(unique_chorms[0][0], pop_size - len(unique_chorms), 0.3)
            selected_chroms.extend(pop_appendix.tolist())
        else:
            logging.info(f'initialize {pop_size} init chroms...  [successful]')
            selected_chroms.extend([unique_chorms[i][0] for i in range(pop_size)])

        return np.array(selected_chroms)

    def generate_by_ascending_distribution(self, data, segments_num, pop_size, repeat_num=50) -> np.ndarray:
        logging.info('choosing the seed chroms with the Inverted Triangle Distribution (IT)')
        logging.info(f'segments_num: {segments_num}')

        init_chroms = []
        for lift_ratio in np.linspace(1.1, 1.5, 10):
            row_cdf = [expon(scale=1 / np.log(lift_ratio)).cdf(xi) for xi in range(1, segments_num + 1)]
            expected_cdf = [cdf / row_cdf[-1] for cdf in row_cdf[:segments_num - 1]]
            expected_cdf = [1 - expected_cdf[len(expected_cdf) - i - 1] for i in range(len(expected_cdf))]
            for k in range(repeat_num):
                possible_bin, actual_cdf = self.generate_possible_bins(data, segments_num, expected_cdf)
                if possible_bin is None or actual_cdf is None:
                    continue
                chrom_score = self.get_chrom_score(expected_cdf, actual_cdf, self.init_strategy)
                init_chroms.append([possible_bin, chrom_score])

        unique_chorms = []
        for chrom in init_chroms:
            if chrom not in unique_chorms:
                unique_chorms.append(chrom)
        unique_chorms.sort(key=lambda x: x[-1])
        selected_chroms = []

        if len(unique_chorms) == 0:
            logging.info(f'failed to initialize any valuable init chroms... [random generation]')
            selected_chroms = self.random_chroms(data, segments_num, pop_size)
        elif len(unique_chorms) < pop_size:
            logging.info(f'initialize {len(unique_chorms)} init chroms (expected {pop_size})...  [disturb chroms]')
            selected_chroms.extend([unique_chorms[i][0] for i in range(len(unique_chorms))])
            pop_appendix = self.disturb_chrom(unique_chorms[0][0], pop_size - len(unique_chorms), 0.3)
            selected_chroms.extend(pop_appendix.tolist())
        else:
            logging.info(f'initialize {pop_size} init chroms...  [successful]')
            selected_chroms.extend([unique_chorms[i][0] for i in range(pop_size)])

        return np.array(selected_chroms)

    def generate_by_normal_distribution(self, data, segments_num, pop_size, repeat_num=50) -> np.ndarray:
        logging.info('choosing the seed chrom with the Normal Distribution (NF)')
        logging.info(f'segments_num: {segments_num}')

        init_chroms = []
        cure_rate = np.cumsum(data) / sum(data)
        prebins_num = len(data)

        p1_max = 1 / segments_num
        for j in range(prebins_num):
            p_first = cure_rate[j]
            if p_first <= p1_max:
                for i in range(prebins_num - 1, j, -1):
                    if 1 - cure_rate[i] > p_first + 1e-8:
                        p_last = 0
                        if i == prebins_num - 1:
                            p_last = cure_rate[i]
                        else:
                            p_last = cure_rate[i + 1]
                        st = norm.ppf(p_first)
                        ed = norm.ppf(p_last)
                        xi = np.linspace(st, ed, segments_num - 1)
                        expected_cdf = norm.cdf(xi)
                        for k in range(repeat_num):
                            possible_bin, actual_cdf = self.generate_possible_bins(data, segments_num, expected_cdf)
                            chrom_score = self.get_chrom_score(expected_cdf, actual_cdf, self.init_strategy)
                            init_chroms.append([possible_bin, chrom_score])
                        break
            else:
                break

        unique_chorms = []
        for chrom in init_chroms:
            if chrom not in unique_chorms:
                unique_chorms.append(chrom)
        unique_chorms.sort(key=lambda x: x[-1])
        selected_chroms = []

        if len(unique_chorms) == 0:
            logging.info(f'failed to initialize any valuable init chroms... [random generation]')
            selected_chroms = self.random_chroms(data, segments_num, pop_size)
        elif len(unique_chorms) < pop_size:
            logging.info(f'initialize {len(unique_chorms)} init chroms (expected {pop_size})...  [disturb chroms]')
            selected_chroms.extend([unique_chorms[i][0] for i in range(len(unique_chorms))])
            pop_appendix = self.disturb_chrom(unique_chorms[0][0], pop_size - len(unique_chorms), 0.3)
            selected_chroms.extend(pop_appendix.tolist())
        else:
            logging.info(f'initialize {pop_size} init chroms...  [successful]')
            selected_chroms.extend([unique_chorms[i][0] for i in range(pop_size)])

        return np.array(selected_chroms)

    def random_chroms(self, data, segments_num, pop_size):
        prebins_num = len(data)
        chrom = [int(i/segments_num * (prebins_num-1)) for i in range(1, segments_num+1)]
        logging.info(f'seed chrom: {chrom}')
        chroms = []
        pop_appendix = self.disturb_chrom(chrom, pop_size-1, 0.3)
        chroms.extend(pop_appendix.tolist())

        return np.array(chroms)


class RangeMutator(ea.Mutation):
    def __init__(self, Pm=0.1, lb=1, ub=100):
        self.Pm = Pm
        self.lb = lb
        self.ub = ub
        super().__init__()

    def do(self, OldChrom):
        chrom_size = len(OldChrom)
        if chrom_size <= 0:
            raise Exception('The population size is zero!')
        chrom_len = len(OldChrom[0])
        offSpring = []

        for i in range(chrom_size):
            o1 = cp.copy(OldChrom[i])
            for j in range(chrom_len - 1):
                rand = np.random.rand()
                if rand <= self.Pm:
                    lb = self.lb - 1 if j == 0 else o1[j - 1]
                    ub = self.ub + 1 if j == chrom_len - 1 else o1[j + 1]
                    if lb + 1 < ub:
                        rand2 = np.random.randint(lb + 1, ub)
                        candidate_set = set(range(lb + 1, ub)) - {o1[j]}
                        if len(candidate_set) == 0:
                            o1[j] = lb + 1
                        else:
                            o1[j] = np.random.choice(list(candidate_set))
            offSpring.append(o1)

        return np.array(offSpring)