import torch.multiprocessing as mp
from model import ppo_clip
import os

class worker(mp.Process):
    def __init__(self, env_id, model_epoch, epoch, max_epoch, population, score_info, capacity=10000, update_iter=3, gamma=0.98, lam=0.95, learning_rate=1e-3, epsilon=0.1):
        super(worker, self).__init__()
        self.env_id = env_id
        self.model_epoch = model_epoch
        self.epoch = epoch
        self.max_epoch = max_epoch
        self.population = population
        self.score_info = score_info
        self.capacity = capacity
        self.update_iter = update_iter
        self.gamma = gamma
        self.lam = lam
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.model = ppo_clip(
            env_id=self.env_id,
            epoch = self.model_epoch,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            lam=self.lam,
            epsilon=self.epsilon,
            capacity=self.capacity,
            update_iter=self.update_iter,
        )

    def run(self):
        while True:
            self.p = self.population.get()
            if self.epoch.value > self.max_epoch:
                break
            self.model.model_id = self.p['id']
            model_path = './models/model_{}.pkl'.format(self.p['id'])
            if os.path.exists(model_path):
                self.model.load_weight_hyperparam(model_path)
            self.model.run()
            score = self.model.eval()
            self.model.save_weight_hyperparam(model_path)
            self.model.reset()
            self.score_info.put(dict(id=self.p['id'], score=score))
