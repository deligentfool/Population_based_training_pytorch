import torch.multiprocessing as mp
import os
from worker import worker
import numpy as np
import torch

class explorer(mp.Process):
    def __init__(self, epoch, max_epoch, population, score_info, fraction, perturb_factors):
        super(explorer, self).__init__()
        self.epoch = epoch
        self.max_epoch = max_epoch
        self.population = population
        self.score_info = score_info
        self.fraction = fraction
        self.perturb_factors = perturb_factors

    def run(self):
        while True:
            if self.epoch.value > self.max_epoch:
                break
            if self.population.empty() and self.score_info.full():
                infos = []
                while not self.score_info.empty():
                    infos.append(self.score_info.get())
                infos = sorted(infos, key=lambda x: x['score'], reverse=True)
                print('======================================================================================')
                print('Explore and exploit')
                print('Best score on\tid:{}\tis\t{:.3f}'.format(infos[0]['id'], infos[0]['score']))
                print('Worst score on\tid:{}\tis\t{:.3f}'.format(infos[-1]['id'], infos[-1]['score']))
                print('======================================================================================')
                update_part_num = int(np.ceil(self.fraction * len(infos)))
                tops = infos[:update_part_num]
                bottoms = infos[-update_part_num:]
                for bottom in bottoms:
                    top = np.random.choice(tops)
                    top_model_path = './models/model_{}.pkl'.format(top['id'])
                    bot_model_path = './models/model_{}.pkl'.format(bottom['id'])

                    top_model = torch.load(top_model_path)
                    policy_weight = top_model['policy_weight']
                    value_weight = top_model['value_weight']
                    hyperparameters = top_model['hyperparameters']
                    for hyperparam_name in hyperparameters.keys():
                        perturb = np.random.choice(self.perturb_factors)
                        hyperparameters[hyperparam_name] *= perturb
                    new_model_ = {}
                    new_model_['policy_weight'] = policy_weight
                    new_model_['value_weight'] = value_weight
                    new_model_['hyperparameters'] = hyperparameters
                    torch.save(new_model_, bot_model_path)
                    with self.epoch.get_lock():
                        self.epoch.value += 1
                for info in infos:
                    self.population.put(info)