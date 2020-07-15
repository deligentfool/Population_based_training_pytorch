from worker import worker
from explorer import explorer
import torch
import numpy as np
import torch.multiprocessing as mp
import os


if __name__ == '__main__':
    population_size = 20
    max_epoch = 10000
    model_epoch = 10
    env_id = 'CartPole-v1'
    # * the fraction's range is (0., 0.5]
    fraction = 0.4
    perturb_factors = np.linspace(0.9, 1.1, 10).tolist()

    os.makedirs('./models/', exist_ok=True)
    population = mp.Queue(maxsize=population_size)
    score_info = mp.Queue(maxsize=population_size)
    epoch = mp.Value('i', 0)
    for i in range(population_size):
        population.put(dict(id=i, score=0))
    workers = [worker(env_id, model_epoch=model_epoch, epoch=epoch, max_epoch=max_epoch, population=population, score_info=score_info) for i in range(mp.cpu_count() - 1)]
    workers.append(explorer(epoch, max_epoch, population, score_info, fraction, perturb_factors))

    [w.start() for w in workers]
    [w.join() for w in workers]

    infos = []
    while not score_info.empty():
        infos.append(score_info.get())
    while not population.empty():
        infos.append(population.get())

    infos = sorted(infos, key=lambda x: x['score'], reverse=True)
    print('Finish! Best score on\tid:{}\tis\t{:.3f}'.format(infos[0]['id'], infos[0]['score']))