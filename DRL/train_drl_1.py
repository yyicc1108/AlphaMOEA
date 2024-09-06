import os
import time
import geatpy as ea
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
torch.set_default_dtype(torch.float64)
import warnings
warnings.filterwarnings("ignore")

from LibMTL import Trainer
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
from LibMTL.loss import MSELoss
import torch.utils.data as Data
from torch.nn.parameter import Parameter
# from torch_geometric.loader import DataLoader
from LibMTL.metrics import L1Metric

from DRL.env import AlphaEAEnv
from problem import ZDT, DTLZ
from operators import evolution
from pymoo.indicators.hv import HV
from paretoset import paretoset
from DRL.PPO import PPO

from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from net import Embedding

import random

moeas = ["nsga2", "moead", "rvea"]

zdt_instances = ["zdt1",
                 "zdt2",
                 # "zdt3",
                 # "zdt4",
                 #"zdt6"
                 ]
dtlz_instances = [
    #"dtlz1",
    #"dtlz2", "dtlz3", "dtlz4",
    "dtlz5", "dtlz6","dtlz7"
                  ]

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

class ProblemTestDataset(Dataset):

    def __init__(self, size=50, num_samples=10000, label=None, mop_instance=None):
        super(ProblemTestDataset, self).__init__()

        all_test_files = get_all_files("../test_data")
        if len(all_test_files) == 0:
            self.dataset = torch.rand(num_samples, size, dtype=torch.float64)  # 生成0-1之间的随机数
        else:
            #print("../test_data/"+mop_instance+"_test.csv")
            data = np.array(pd.read_csv("../test_data/"+mop_instance+"_test.csv", header=0))
            self.dataset = torch.from_numpy(data)
        self.size = len(self.dataset)
        self.label = {}
        label = torch.from_numpy(np.array(label))
        for tn in range(label.shape[0]):
            self.label[str(tn)] = label[tn, :]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.dataset[idx, :], self.label
def get_all_file_with_e_s_str(dir, e_str='.csv', s_str=None):
    file_list = []
    for root_dir, sub_dir, files in os.walk(r'' + dir):
        # 对文件列表中的每一个文件进行处理，如果文件名字是以‘xlxs’结尾就
        # 认定为是一个excel文件，当然这里还可以用其他手段判断，比如你的excel
        # 文件名中均包含‘results’，那么if条件可以改写为
        for file in files:
            # if file.endswith('.xlsx') and 'results' in file:
            if file.endswith(e_str) and file.startswith(s_str):
                # 此处因为要获取文件路径，比如要把D:/myExcel 和res.xlsx拼接为
                # D:/myExcel/results.xlsx，因此中间需要添加/。python提供了专门的
                # 方法
                file_name = os.path.join(root_dir, file)
                # 把拼接好的文件目录信息添加到列表中
                file_list.append(file_name)
    return file_list
def get_all_files(dir, ):
    file_list = []
    for root_dir, sub_dir, files in os.walk(r'' + dir):
        # 对文件列表中的每一个文件进行处理，如果文件名字是以‘xlxs’结尾就
        # 认定为是一个excel文件，当然这里还可以用其他手段判断，比如你的excel
        # 文件名中均包含‘results’，那么if条件可以改写为
        for file in files:
            # if file.endswith('.xlsx') and 'results' in file:
            if file.endswith('.xlsx') or file.endswith('.csv'):
                # 此处因为要获取文件路径，比如要把D:/myExcel 和res.xlsx拼接为
                # D:/myExcel/results.xlsx，因此中间需要添加/。python提供了专门的
                # 方法
                file_name = os.path.join(root_dir, file)
                # 把拼接好的文件目录信息添加到列表中
                file_list.append(file_name)
    return file_list
def init(mop_instance):
    train_data_files = get_all_files(dir='../training_data/Part-1-2')
    for file in train_data_files:
        if mop_instance in file:
            label_data = pd.read_csv(file, header=0)

    return label_data


def get_all_PFs_scaler(instance):

    all_files = []
    for moea in moeas:
        all_files.append(get_all_file_with_e_s_str('../results/Part-1-2/' + instance[:-1], s_str=moea + '_' + instance, e_str='csv'))

    n_obj = 2 if instance[0] == 'z' else 3
    all_PFs = np.zeros((1,n_obj))

    for ind, files in enumerate(all_files):
        for file in files:
            pf = pd.read_csv(file, header=0)
            # if ind==3:
            #     pf = abs(pf)
            all_PFs = np.vstack((all_PFs, pf))

    #print(all_PFs)
    all_PFs = all_PFs[1:, :]
    scaler = MinMaxScaler()
    scaler = scaler.fit(all_PFs)

    return scaler

class MOPEncoder(nn.Module):
    def __init__(self, emb_dim, nhead):
        # =============================================================================
        # 建立模型, 由三层transformer Encoder layer 与三层全连接网络构成，尾部添加softmax
        # =============================================================================
        super(MOPEncoder, self).__init__()
        self.linear1 = nn.Linear(1, emb_dim)
        self.transform_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=8 * emb_dim,
                                                          batch_first=True)
        self.transform_encoder = nn.TransformerEncoder(self.transform_layer, num_layers=1)

    def forward(self, X):
        # =============================================================================
        #         X:输入，维度为(batch_size, seq_length)
        #         max_length:最大缺陷点个数(不足的padding 0矩阵)
        #         seq_length:输入序列长度
        # =============================================================================
        X = X.unsqueeze(2)  # 扩展维度 (batch_size, seq_length) -> (batch_size, seq_length, embedding dimension)
        X = F.relu(self.linear1(X))
        out = self.transform_encoder(X).view(X.size(0), -1)  # (batch_size, seq_length*embedding dimension)
        return out

def main(params, label_data, problem, seq_len, target, nhead, emb_dim, cuda):
    train_dataset = problem.make_dataset(size=seq_len, num_samples=100000, label=label_data)
    test_dataset = problem.make_dataset(size=seq_len, num_samples=5000, label=label_data)
    val_dataset = problem.make_dataset(size=seq_len, num_samples=5000, label=label_data)
    train_loader = Data.DataLoader(train_dataset, batch_size=params.bs, pin_memory=True, )
    test_loader = Data.DataLoader(test_dataset, batch_size=params.bs, pin_memory=True, )
    val_loader = Data.DataLoader(val_dataset, batch_size=params.bs, pin_memory=True, )

    kwargs, optim_param, scheduler_param = prepare_args(params)

    scheduler_param = {'scheduler': 'reduce',
                       'mode': 'max',
                       'factor': 0.7,
                       'patience': 5,
                       'min_lr': 0.00001}

    # target = params.target

    device = torch.device(cuda)

    # define tasks
    task_dict = {}
    for _, t in enumerate(target):
        task_dict[str(t)] = {'metrics': ['MAE'],
                             'metrics_fn': L1Metric(),
                             'loss_fn': MSELoss(),
                             'weight': [0]}

    # define encoder and decoder
    class MOPEncoder(nn.Module):
        def __init__(self):
            # =============================================================================
            # 建立模型, 由三层transformer Encoder layer 与三层全连接网络构成，尾部添加softmax
            # =============================================================================
            super(MOPEncoder, self).__init__()
            self.linear1 = nn.Linear(1, emb_dim)
            self.transform_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=8 * emb_dim,
                                                              batch_first=True)
            self.transform_encoder = nn.TransformerEncoder(self.transform_layer, num_layers=1)

        def forward(self, X):
            # =============================================================================
            #         X:输入，维度为(batch_size, seq_length)
            #         max_length:最大缺陷点个数(不足的padding 0矩阵)
            #         seq_length:输入序列长度
            # =============================================================================
            X = X.unsqueeze(2)  # 扩展维度 (batch_size, seq_length) -> (batch_size, seq_length, embedding dimension)
            X = F.relu(self.linear1(X))
            out = self.transform_encoder(X).view(X.size(0), -1)  # (batch_size, seq_length*embedding dimension)
            return out

    def encoder_class():
        return MOPEncoder()

    MOPdecoders = nn.ModuleDict({task: nn.Linear(seq_len * emb_dim, seq_len) for task in list(task_dict.keys())})

    class MOPtrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class,
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, device, **kwargs):
            super(MOPtrainer, self).__init__(task_dict=task_dict,
                                             weighting=weighting_method.__dict__[weighting],
                                             architecture=architecture_method.__dict__[architecture],
                                             encoder_class=encoder_class,
                                             decoders=MOPdecoders,
                                             rep_grad=rep_grad,
                                             multi_input=multi_input,
                                             optim_param=optim_param,
                                             scheduler_param=scheduler_param,
                                             device=device,
                                             **kwargs)

    MOPmodel = MOPtrainer(task_dict=task_dict,
                          weighting=params.weighting,
                          architecture=params.arch,
                          encoder_class=encoder_class,
                          decoders=MOPdecoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          device=device,
                          **kwargs)
    #MOPmodel.train(train_loader, test_loader, 100, val_dataloaders=val_loader)

    return MOPmodel

def get_init_PF(instance):

    all_files = get_all_file_with_e_s_str('../results/'+instance[:-1], s_str='alphaEA_'+instance, e_str='csv')

    problem = Problems[instance]

    for index, files in enumerate(all_files):
        if index == 0:
            PF = pd.read_csv(files, header=0)
        else:
            temp = pd.read_csv(files, header=0)
            PF = pd.concat([PF, temp])

    if instance == "zdt6":
        mask = paretoset(PF, sense=["max"] * problem.M)  # 计算PF
    else:
        mask = paretoset(PF, sense=["min"] * problem.M)  # 计算PF

    PF = PF[mask]

    if instance == "dtlz1":
        for col in PF.columns:
            PF = PF[PF[col] < 0.5]

        for col in PF.columns:
            PF = PF[PF[col] > 0]

    return PF
def get_encoder_output(model, instance, emb_dim, nhead):

    input = np.array(pd.read_csv("../test_data/"+instance+"_test.csv", header=0))
    input = torch.from_numpy(input[:30, :])
    encoders = MOPEncoder(emb_dim=emb_dim, nhead=nhead)
    model2_dict = encoders.state_dict()
    layer_names = []
    for name in list(model2_dict.keys()):
        layer_names.append("encoder." + name)
    state_dict = {k[8:]: v for k, v in model.state_dict().items() if k in layer_names}
    model2_dict.update(state_dict)
    encoders.load_state_dict(model2_dict)

    output = encoders(input)

    return output
def parse_args(parser_sl):

    # parameters for SL training process
    parser_sl.add_argument('--bs', default=128, type=int, help='batch size')
    # parser.add_argument('--seq_len', default=label_data.shape[1], type=int, help='sequence len')
    parser_sl.add_argument('--embed_dim', default=64, type=int, help='embedding dimension')
    parser_sl.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    # parser.add_argument('--target', default=[*range(label_data.shape[0])], type=int, nargs='+', help='target')

    # parameters for DRL training process
    parser_rl = argparse.ArgumentParser(description="hyper parameters")
    parser_rl.add_argument('--env_name', default='ALPHA-MOEA', type=str, help="name of environment")
    parser_rl.add_argument('--run_num', default=1, type=int, help="total number of running")
    parser_rl.add_argument('--n_epochs', default=int(1e3), type=int, help='max training epochs for each run')
    parser_rl.add_argument('--print_freq', default=100, type=int, help='print avg reward in the interval')
    parser_rl.add_argument('--plot_freq', default=100, type=int, help='print avg reward in the interval')
    parser_rl.add_argument('--update_epoch', default=50, type=int, help='print avg reward in the interval')
    parser_rl.add_argument('--K_epochs', default=10, type=int, help='update policy for K epochs in one PPO update')
    parser_rl.add_argument('--random_seed', default=0, type=int, help='set random seed if required (0 = no random seed)')
    parser_rl.add_argument('--hidden_dim', default=64, type=int, help='set random seed if required (0 = no random seed)')
    parser_rl.add_argument('--eps_clip', default=0.2, type=float, help="clip parameter for PPO")
    parser_rl.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser_rl.add_argument('--lr_actor', default=0.0003, type=float, help="learning rate of actor net")
    parser_rl.add_argument('--lr_critic', default=0.001, type=float, help="learning rate of critic net")

    return parser_sl.parse_args(), parser_rl.parse_args()
def load_SL_model(instances, params_sl, params_rl, nhead, emb_dim, cuda):

    times = []
    test_times = []
    for mop_instance in instances:

        with torch.no_grad():

            label_data = init(mop_instance)
            target = [*range(label_data.shape[0])]
            seq_len = label_data.shape[1]

            problem_pymoo = DTLZ(problem_name=mop_instance, size=seq_len) if mop_instance[:4]=='dtlz' else ZDT(problem_name=mop_instance, size=seq_len)
            problem_ea = Problems[mop_instance]

            test_dataset = ProblemTestDataset(size=seq_len, num_samples=3000, label=label_data, mop_instance=mop_instance)
            test_loader = Data.DataLoader(test_dataset, batch_size=1, shuffle=False)

            scaler = get_all_PFs_scaler(mop_instance)

            #set_random_seed(params_sl.seed)
            MOPmodel = main(params_sl, label_data, problem_pymoo, seq_len, target, nhead, emb_dim, cuda)
            model_save_path = os.path.join('../trained_model/SL/', mop_instance + '.pt')
            MOPmodel.model.load_state_dict(torch.load(model_save_path, map_location='cpu'))
            #MOPmodel.model.load_state_dict(torch.load(model_save_path))

            #init_PF = get_init_PF(instance=mop_instance)
            encoder_output = get_encoder_output(model=MOPmodel.model, instance=mop_instance, emb_dim=emb_dim, nhead=nhead)

            env = AlphaEAEnv(mop_instance, MOPmodel.model, encoder_output, problem_ea, seq_len, emb_dim
                             , plot_freq = params_rl.plot_freq
                             , plot_epoch = params_rl.n_epochs
                             , scaler = scaler
                             # , Nsample= params_rl.Nsample
                             # , max_iters= params_rl.max_iters
                             # , F1=params_rl.F1
                             # , F2=params_rl.F2
                             )

            weights, PF, r_list, a_list, t= train_alphaEA_PPO(env, params_rl)
            print("training time:", t)
            decoders = [ nn.Linear(seq_len * emb_dim, seq_len) for _ in range(len(weights))]
            for i, weight in enumerate(weights):
                decoders[i].weight = Parameter(torch.from_numpy(weight[:, :-1]))
                decoders[i].bias = Parameter(torch.from_numpy(weight[:, -1]))

            MOPdecoders = nn.ModuleDict({str(task): decoders[task] for task in [*range(len(weights))]})
            MOPmodel.model.decoders.update(MOPdecoders)

            s_time = time.time()
            _ = MOPmodel.test(test_dataloaders=test_loader, get_res=True)
            test_times.append(time.time() - s_time)
            print(test_times)

            model_save_path = os.path.join('../trained_model/DRL/', mop_instance + '.pt')
            torch.save(MOPmodel.model.state_dict(), model_save_path)
            times.append(t)

            pd.DataFrame(np.array(r_list).reshape(-1, 100)).to_csv('../results/drl/' + mop_instance.upper() + '-AlphaEA-' + '-rewards.csv', index=False)
            pd.DataFrame(np.array(a_list).reshape(-1, 100)).to_csv('../results/drl/' + mop_instance.upper() + '-AlphaEA-' + '-actions.csv', index=False)
            pd.DataFrame(PF).to_csv('../results/drl/' + mop_instance.upper() + '-AlphaEA-' + '-PF.csv', index=False)
            # CCA process
            # for name, m in MOPmodel.model.named_modules():  # 返回的每一个元素是一个元组 tuple
            #     '''
            #     是一个元组 tuple ,元组的第一个元素是参数所对应的名称，第二个元素就是对应的参数值
            #     '''
            #     if isinstance(m, (nn.Linear, nn.Conv2d)):
            #         if name[:2] == 'de':
            #             decoders_weight_list.append(m.weight.data.reshape(-1, 1).numpy())
            #         print(name, '\t', m.weight.size())
            #
            #
            # init_state = np.ones((len(decoders_weight_list), len(decoders_weight_list)))
            # for i in range(0, len(decoders_weight_list)):
            #     for j in range(i+1, len(decoders_weight_list)):
            #         vec1 = decoders_weight_list[i]
            #         vec2 = decoders_weight_list[j]
            #         ca = CCA(n_components=1)
            #         ca.fit(vec1, vec2)
            #         X_c, Y_c = ca.transform(vec1, vec2)
            #         cos_sim = np.corrcoef(X_c[:, 0], Y_c[:, 0])
            #         init_state[i][j], init_state[j][i] = cos_sim[0][1], cos_sim[0][1]
    pd.DataFrame(test_times).to_csv('../results/times/' + mop_instance[:-1].upper() + '-AlphaEA-' + 'drl-test-times.csv',index=False)
    pd.DataFrame(times).to_csv('../results/times/alpha_' + mop_instance[:-1] + '_DRL_training_times.csv', index=False)
def train_alphaEA_PPO(env, params_rl):

    #torch.manual_seed(params_rl.random_seed)
    #env.seed(params_rl.random_seed)
    #np.random.seed(params_rl.random_seed)

    # state and action space dimension
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ############# print all hyperparameters #############
    print("============================================================================================")
    print("training environment name : " + params_rl.env_name)
    #print("total number of runs : ", params_rl.run_num)
    print("max training episodes : ", params_rl.n_epochs)
    print("printing average reward over episodes in last : " + str(params_rl.print_freq) + " epochs")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("PPO update frequency : " + str(params_rl.update_epoch) + " epochs")
    print("PPO K epochs : ", params_rl.K_epochs)
    print("PPO hidden dimension : ", params_rl.hidden_dim)
    print("PPO epsilon clip : ", params_rl.eps_clip)
    print("discount factor (gamma) : ", params_rl.gamma)
    print("optimizer learning rate actor : ", params_rl.lr_actor)
    print("optimizer learning rate critic : ", params_rl.lr_critic)

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, params_rl.hidden_dim, params_rl.lr_actor, params_rl.lr_critic, params_rl.gamma, params_rl.K_epochs, params_rl.eps_clip)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    s_time = time.time()
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    i_episode = 0
    r_list = []
    # training loop
    state = env.reset()
    a_list = []
    env.plot(0)
    while i_episode < params_rl.n_epochs:
        # select action with policy
        action = ppo_agent.select_action(state)
        state, reward, done, _ = env.step(action)
        a_list.append(action)
        r_list.append(reward)

        # saving reward and is_terminals
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)

        # update PPO agent
        if (i_episode+1) % params_rl.update_epoch == 0:
            ppo_agent.update()

        # printing average reward
        if (i_episode+1) % params_rl.print_freq == 0:

            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            #print_avg_reward = round(print_avg_reward, 2)

            print("Episode : {} \t\t Average Reward : {} \t\t Times : {}".format(i_episode+1, print_avg_reward, datetime.now().replace(microsecond=0)-start_time))


            print_running_reward = 0
            print_running_episodes = 0
        if (i_episode+1) % params_rl.plot_freq == 0:
            env.plot(i_episode)


        # break; if the episode is over
        if (time.time()-s_time) > 7200:
            break

        print_running_reward += reward
        print_running_episodes += 1

        i_episode += 1

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    return env.weights, env.PF, r_list, a_list, end_time - start_time


if __name__ == "__main__":

    # generate_test_data()

    cuda = 'cpu'
    nheads = [2, 4, 8]
    emb_dims = [64, 128, 256]

    params_sl, params_rl = parse_args(LibMTL_args)
    load_SL_model(dtlz_instances, params_sl, params_rl, nheads[0], emb_dims[0], cuda)
    #load_SL_model(zdt_instances, params_sl, params_rl, nheads[2], emb_dims[2], cuda)
    #test_alphaEA_model(zdt_instances, params)




