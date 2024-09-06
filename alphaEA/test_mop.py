import os
import time
import numpy as np
import pandas as pd
import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
torch.set_default_tensor_type(torch.DoubleTensor)

from LibMTL import Trainer
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
from LibMTL.loss import MSELoss
import torch.utils.data as Data
# from torch_geometric.loader import DataLoader
from LibMTL.metrics import L1Metric

from problem import ZDT, DTLZ

from pymoo.indicators.hv import HV
from paretoset import paretoset

zdt_instances = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"]
#zdt_instances = ["zdt6"]
#dtlz_instances = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7"]
dtlz_instances = [ "dtlz6"]


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
    train_data_files = get_all_files(dir='../training_data')
    for file in train_data_files:
        if mop_instance in file:
            label_data = pd.read_csv(file, header=0)

    return label_data


def parse_args(parser):
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    # parser.add_argument('--seq_len', default=label_data.shape[1], type=int, help='sequence len')
    parser.add_argument('--embed_dim', default=64, type=int, help='embedding dimension')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    # parser.add_argument('--target', default=[*range(label_data.shape[0])], type=int, nargs='+', help='target')
    return parser.parse_args()

def main(params, label_data, problem, seq_len, target, nhead=2, emb_dim=64):
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
            self.transform_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead
                                                              #, dim_feedforward=4*emb_dim
                                                              , batch_first=True)
            self.transform_encoder = nn.TransformerEncoder(self.transform_layer, num_layers=2)

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

    MOPdecoders = nn.ModuleDict(
        {task: nn.Linear(seq_len * emb_dim, seq_len) for task in list(task_dict.keys())})
    device=torch.device('cpu')
    class MOPtrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class,
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, device,**kwargs):
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

def test_alphaEA_model(instances, params):

    test_times = []
    for mop_instance in instances:
        label_data = init(mop_instance)
        target = [*range(label_data.shape[0])]
        seq_len = label_data.shape[1]

        problem = DTLZ(problem_name=mop_instance, size=seq_len) if mop_instance[:4]=='dtlz' else ZDT(problem_name=mop_instance, size=seq_len)

        test_dataset = ProblemTestDataset(size=seq_len, num_samples=3000, label=label_data, mop_instance=mop_instance)
        test_loader = Data.DataLoader(test_dataset, batch_size=100, shuffle=False)

        set_random_seed(params.seed)
        MOPmodel = main(params, label_data, problem, seq_len, target)
        model_save_path = os.path.join('../trained_model/SL/', mop_instance + '.pt')
        MOPmodel.model.load_state_dict(torch.load(model_save_path, map_location='cpu'))

        s_time = time.time()
        results = MOPmodel.test(test_dataloaders=test_loader, get_res=True)
        test_times.append(time.time() - s_time)

        for index in range(len(results)):
            ps_set = np.array([value.numpy() for key, value in results[index].items()]).reshape(-1, seq_len)
            sols = problem.get_costs(ps_set)
            sols = sols[~np.isnan(sols).any(axis=1)]
            if mop_instance == "zdt6":
                df = pd.DataFrame(sols)
                for col in df.columns:
                    df = df[df[col] < 1]
                sols = np.array(df)
                #sols = np.delete(sols, np.where((sols >= 1)), axis=0)
                mask = paretoset(sols, sense=["max"] * problem.problem.n_obj)  # 计算PF
            else:
                mask = paretoset(sols, sense=["min"] * problem.problem.n_obj)  # 计算PF
            PF = sols[mask]
            pd.DataFrame(PF).to_csv('../results/' + mop_instance[:-1] + '/alphaEA_' + mop_instance + '_' + str(index + 1) + '.csv', index=False)

    pd.DataFrame(test_times).to_csv('../results/times/alpha_' + mop_instance[:-1] + '_time.csv', index=False)

if __name__ == "__main__":

    # generate_test_data()
    params = parse_args(LibMTL_args)
    test_alphaEA_model(dtlz_instances, params)
    #test_alphaEA_model(zdt_instances, params)




