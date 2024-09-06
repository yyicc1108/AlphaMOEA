import gym
import numpy as np
import pandas as pd
import geatpy as ea
import torch
import random
from gym import spaces
import torch.nn as nn

from torch.nn.parameter import Parameter
from paretoset import paretoset
from operators import evolution
from pymoo.indicators.hv import HV

from itertools import compress

Problems = {
    'dtlz1': ea.benchmarks.DTLZ1(),
    'dtlz2': ea.benchmarks.DTLZ2(),
    'dtlz3': ea.benchmarks.DTLZ3(),
    'dtlz4': ea.benchmarks.DTLZ4(),
    'dtlz5': ea.benchmarks.DTLZ5(),
    'dtlz6': ea.benchmarks.DTLZ6(),
    'dtlz7': ea.benchmarks.DTLZ7(),
    'zdt1': ea.benchmarks.ZDT1(),
    'zdt2': ea.benchmarks.ZDT2(),
    'zdt3': ea.benchmarks.ZDT3(),
    'zdt4': ea.benchmarks.ZDT4(),
    'zdt5': ea.benchmarks.ZDT5(),
    'zdt6': ea.benchmarks.ZDT6(),
            }

def drop_close_data(PF):

    threshold = 0.0001
    diff = np.empty(PF.shape)
    diff[0] = np.inf  # always retain the 1st element
    diff[1:, :] = np.diff(PF, axis=0)
    mask = diff.sum(axis=1) > threshold

    PF = PF[mask]

    return PF

class AlphaEAEnv(gym.Env):

    """Custom Environment that follows gym interface"""

    def __init__(self, instance, model, encoder_output, problem_ea, seq_len, hidden_dim, plot_freq, plot_epoch, scaler, Nsample=100, max_iters=500, F1=0.0001, F2=0.0001):
        super(AlphaEAEnv, self).__init__()

        self.weigh_list = self._get_weights_from_model(model)
        self.plot_freq = plot_freq
        self.plot_epoch = plot_epoch
        self.scaler = scaler


        self.F1 = F1
        self.F2 = F2
        self.problem = problem_ea
        self.max_iters = max_iters
        self.instance = instance
        self.Nsample = Nsample
        self.encoder_output = encoder_output
        #self.max_in_seq_len = (Nsample * (Nsample-1)) // 2
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.Nsample, 1), dtype=np.float32)

        self.evo = evolution(M=problem_ea.M)
        self.decoder = nn.Linear(seq_len * hidden_dim, seq_len)

        #self.sols, self.PF = self._get_sols_PF(self.weigh_list)
        self.PF, self.weights = self._get_sols_PF(self.weigh_list)
        self.init_PF = self.PF
        self.cnt = 0

        self.hv = HV(ref_point=np.array([1.0] * self.problem.M))

    def _step(self, action):

        if action == 0:
            new_weigh_list = self.evo.operator(pops=self.weigh_list, action=0)
        elif action == 1:
            new_weigh_list = self.evo.operator(pops=self.weigh_list, action=1)
        else:
            new_weigh_list = self.evo.operator(pops=self.weigh_list, action=2)

        # evaluate
        new_sols, new_PF = self._get_sols_PF(new_weigh_list)
        PF = np.vstack([self.PF, new_PF])
        mask = paretoset(PF, sense=["min"] * self.problem.M)  # 计算PF
        PF = PF[mask]
        PF = self.process_PF(PF)

        flag1 = [int((PF == sol).all(1).any()) for sol in self.sols]
        flag2 = [int((PF == sol).all(1).any()) for sol in new_sols]
        print(np.sum(flag1))
        print(np.sum(flag2))
        print(np.sum(flag1)+np.sum(flag2))
        ratio1 = [np.sum(flag1[i]) for i in range(0, len(flag1), 1)]      # 计算贡献率
        ratio2 = [np.sum(flag2[i]) for i in range(0, len(flag2), 1)]      # 计算贡献率
        index1 = [i for i,j in enumerate(ratio1) if j == 0]
        index2 = [i for i,j in enumerate(ratio2) if j == 0]

        # for next state
        self.weigh_list = [n for i, n in enumerate(self.weigh_list) if i not in index1] + [n for i, n in enumerate(new_weigh_list) if i not in index2]
        weights = [weight.ravel() for weight in self.weigh_list]
        observation = self._get_observation(weights)

        # for reward
        reward = np.sum(ratio2)

        # update for next step
        # temp1 = np.delete(new_sols.reshape(-1, 100, self.problem.M), index1, axis=0)
        # temp2 = np.delete(self.sols.reshape(-1, 100, self.problem.M), index2, axis=0)
        # temp1 = np.delete(new_sols, index1, axis=0)
        # temp2 = np.delete(self.sols, index2, axis=0)
        #self.sols = np.vstack([temp1, temp2]).reshape(-1, self.problem.M)
        self.sols = PF
        self.PF = PF

        if self.cnt % 5 ==0:
            self._plot(PF)

        if self.cnt < 200:
            self.cnt += 1
            done = False
        else:
            done = True

        info= {}

        return observation, reward, done, info

    def step(self, action):

        if action == 0:
            new_weigh_list = self.evo.operator(pops=self.weights, pf=self.PF, action=0)
        elif action == 1:
            new_weigh_list = self.evo.operator(pops=self.weights, pf=self.PF, action=1)
        # else:
        #     new_weigh_list = self.evo.operator(pops=self.weigh_list, action=2)

        # evaluate
        new_PF, new_weighs = self._get_sols_PF(new_weigh_list)
        PF = np.vstack([self.PF, new_PF])
        total_weights = self.weights+new_weighs
        #PF, del_index = self.process_PF(PF)

        mask = paretoset(PF, sense=["min"] * self.problem.M)  # 计算PF
        PF = PF[mask]
        weights = list(compress(total_weights, mask))

        PF, delete_index = self.process_PF(PF)
        total_weights = [n for i, n in enumerate(weights) if i not in delete_index]

        weight_for_state = [weight.ravel() for weight in total_weights]
        observation = self._get_observation(weight_for_state)

        self.weights = total_weights
        self.PF = PF

        # for reward
        #reward = np.sum([int((PF == pf).all(1).any()) for pf in new_PF])
        reward_ = self.hv(self.scaler.transform(PF))

        assert reward_ < 1.0, "HV > 1.0"

        #
        # # update for next step
        # # temp1 = np.delete(new_sols.reshape(-1, 100, self.problem.M), index1, axis=0)
        # # temp2 = np.delete(self.sols.reshape(-1, 100, self.problem.M), index2, axis=0)
        # # temp1 = np.delete(new_sols, index1, axis=0)
        # # temp2 = np.delete(self.sols, index2, axis=0)
        # #self.sols = np.vstack([temp1, temp2]).reshape(-1, self.problem.M)
        # self.sols = PF
        # self.PF = PF

        # if self.cnt+1 % self.max_iters == 0:
        #     self.cnt += 1
        #     done = False
        # else:
        #     done = True

        done = False
        info= {}
        return observation, reward_, done, info

    def reset(self):

        weights = [weight.ravel() for weight in self.weigh_list]

        observation = self._get_observation(weights)

        return observation

        #return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        ...

    def close(self):
        ...

    def _get_weights_from_model(self, model):

        weight_list = []
        for name, m in model.named_modules():  # 返回的每一个元素是一个元组 tuple
            '''
            是一个元组 tuple ,元组的第一个元素是参数所对应的名称，第二个元素就是对应的参数值
            '''
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if name[:2] == 'de':

                    #weight_list_state.append(np.concatenate((m.weight.data.reshape(-1, 1).numpy().ravel(), m.bias.data)))
                    weight_list.append(np.hstack((m.weight.data.cpu().numpy(), m.bias.data.cpu().numpy().reshape(-1, 1))))
                #print(name, '\t', m.weight.size())

        return weight_list

    def _get_observation(self, weights):

        #print('Weight Count:', len(weights))

        samples = random.choices(weights, k=self.Nsample)  if len(weights) < self.Nsample else random.sample(weights, self.Nsample)# 按照均匀分布抽样Nsample个

        observation = np.zeros((self.Nsample+1))

        #dim = (len(weights) * (len(weights)-1)) // 2
        #observation = np.zeros((dim, 1))
        #observation = np.zeros((self.max_in_seq_len+2, 1))
        for i in range(0, self.Nsample):
            for j in range(i+1, self.Nsample):
                vec1 = samples[i]
                vec2 = samples[j]
                cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                #observation[i][j], observation[j][i] = cos_sim, cos_sim
                observation[j] = cos_sim

        # observation[-1, :,] = len(self.weigh_list) # 记录长度
        observation = observation[1:]  # 删掉第一项

        return observation

    def _get_sols_PF(self, weigh_list):

        #res = np.zeros((len(weigh_list), 100, self.problem.M))
        res = np.zeros((len(weigh_list), self.problem.M))
        for i, weight in enumerate(weigh_list):
            self.decoder.weight = Parameter(torch.from_numpy(weight[:, :-1]))
            self.decoder.bias = Parameter(torch.from_numpy(weight[:, -1]))
            with torch.no_grad():
                sol = self.decoder(self.encoder_output).cpu().numpy().mean(axis=0)
            res[i, :] = self.problem.evalVars(sol.reshape(1, -1))
        sols = res.reshape(-1, self.problem.M)
        sols = np.abs(sols) if self.instance not in ['zdt3'] else sols
        sols = np.delete(sols, np.where((sols >= 1.0))[0], axis=0)  if self.instance not in ['dtlz6', 'dtlz7', 'zdt3'] else sols

        #assert (sols <+ 1.0).all(), "Sols have value >= 1.0"

        mask = paretoset(sols, sense=["min"] * self.problem.M)  # 计算PF
        PF = sols[mask]
        weights = list(compress(weigh_list, mask))

        PF, delete_index = self.process_PF(PF)
        _weights = [n for i, n in enumerate(weights) if i not in delete_index]

        return PF, _weights

    def process_PF(self, PF):

        temp = PF.copy()

        threshold = 0.005
        delete_index = []
        temp = temp[np.argsort(temp[:, 0])]
        for index in range(temp.shape[0]-1):
            diff = np.linalg.norm(temp[index, :] - temp[index+1, :])
            if diff < threshold:
                delete_index.append(index+1)

        temp = np.delete(temp, delete_index, axis=0)

        return temp, delete_index

    def plot(self, epoch=-1):

        PF = self.PF.copy()
        mask = paretoset(np.vstack([self.init_PF, PF]), sense=["min"] * self.problem.M)
        #print(mask[:self.init_PF.shape[0]])# 计算PF
        init_PF = self.init_PF[mask[:self.init_PF.shape[0]], :]

        if self.instance == "dtlz5" or self.instance == "dtlz6":
            view = (25, 10)
        else:
            view = (30, 45)

        if (epoch+1) == 1:

            plotter = ea.PointScatter(self.problem.M, grid=True, legend=False, title=None,
                                      saveName='../plots/drl_training_plots/' + self.instance + '_PFs_' + str(epoch+1), view=view)
            plotter.add(self.problem.ReferObjV, color='gray', alpha=0.1, label='True PF')
            plotter.add(PF, color='red', label='AlphaEA PF')
            plotter.draw()
            plotter.show()

        elif (epoch+1) == self.plot_epoch:

            plotter = ea.PointScatter(self.problem.M, grid=True, legend=False, title=None,
                                      saveName='../plots/drl_training_plots/' + self.instance + '_PFs_' + str(epoch+1), view=view)
            plotter.add(self.problem.ReferObjV, color='gray', alpha=0.1, label='True PF')
            plotter.add(PF, color='red', label='AlphaEA PF')
            plotter.draw()
            plotter.show()

        elif (epoch+1) % self.plot_freq == 0:
            PF = PF[np.argsort(PF[:, 0])]
            init_PF = init_PF[np.argsort(init_PF[:, 0])]

            plotter = ea.PointScatter(self.problem.M, grid=True, legend=False, title=None,
                                      saveName='../plots/drl_training_plots/'+self.instance + '_PFs_' + str(epoch+1), view=view)
            plotter.add(self.problem.ReferObjV, color='gray', alpha=0.1, label='True PF')
            plotter.add(PF, color='green', label='AlphaMOEA')
            if self.instance[0] == 'z':
                plotter.add(init_PF, color='red', alpha=0.5, label='AlphaEA PF')
            else:
                plotter.add(init_PF, color='red', label='AlphaEA PF')
            plotter.draw()
            plotter.show()