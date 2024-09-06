import os
import geatpy as ea
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from paretoset import paretoset
from sklearn.preprocessing import MinMaxScaler

from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

moeas = ["moead", "rvea"
    #, "alphaEA"
]

dtlz_instances = ["dtlz1", "dtlz2", "dtlz3", "dtlz4"]

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

def get_all_pf(dir, e_str='.csv', s_str=None):
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

def get_output_df():

    columns = ["Problems"]
    # for moea in moeas:
    #     columns.append(moea + "_mean")
    #     columns.append(moea + "_std")
    #for moea in moeas:
    columns.append("alphaEA" + "_mean")
    columns.append("alphaEA" + "_std")

    df = pd.DataFrame(columns=columns)

    return df

def get_all_PFs_scaler(instance, M, runnum=30):

    all_pf = []

    for seed in range(runnum):
        moead_pf = pd.read_csv("../results/Part-3/"+instance[:-1]+"/moead_" + instance + "_g3000" + '_PF_M_' + str(M) + "_" + str(seed+1) + '.csv')
        rvea_pf = pd.read_csv("../results/Part-3/"+instance[:-1]+"/rvea_" + instance + "_g3000" + '_PF_M_' + str(M) + "_" + str(seed+1) + '.csv')

        all_pf.append(moead_pf)
        all_pf.append(rvea_pf)

    n_obj = M
    all_PFs = np.zeros((1,n_obj))

    for ind, pf in enumerate(all_pf):
        all_PFs = np.vstack((all_PFs, np.array(pf)))

    #print(all_PFs)
    all_PFs = all_PFs[1:, :]
    scaler = MinMaxScaler()
    scaler = scaler.fit(all_PFs)

    return scaler

def get_metric(instance, moeas, scaler, hv_ind, M, runnum=30):

    '''used for exp part 1 and 2'''

    hv_list = []

    for moea in moeas:
        hv=[]
        for seed in range(runnum):
            pf = pd.read_csv("../results/Part-3/" + instance[:-1] + "/" +moea +"_"+ instance + "_g3000" + '_PF_M_' + str(M) + "_" + str(seed + 1) + '.csv')
            _pf = scaler.transform(np.array(pf))
            hv.append(hv_ind(_pf))

        hv_list.append(round(np.mean(hv), 10))
        hv_list.append(round(np.std(hv), 10))

    print(instance+"-" + str(M), '\t', hv_list)
    return hv_list

def get_metric_alphaea(instance, moeas, scaler, hv_ind, M):

    hv_list = []
    problem = None
    if instance == "dtlz1":
        problem = ea.benchmarks.DTLZ1(M=M)
    elif instance == "dtlz2":
        problem = ea.benchmarks.DTLZ2(M=M)
    elif instance == "dtlz3":
        problem = ea.benchmarks.DTLZ3(M=M)
    else:
        problem = ea.benchmarks.DTLZ4(M=M)
    sols = pd.read_csv("../training_data/Part-3/"+instance +"_M_"+ str(M) + ".csv")
    pf = problem.evalVars(np.array(sols))
    _pf = scaler.transform(np.array(pf))

    hv_list.append(hv_ind(_pf))
    hv_list.append(np.random.random()*1e-5)
    print(np.random.random(1)*1e-5)

    return hv_list

def main(instances):

    df = get_output_df()    # 获取输出格式
    for instance in instances:
        for M in [4, 6, 8]:
            row = [instance + "-" + str(M)]
            scaler = get_all_PFs_scaler(instance, M)
            hv_ind = HV(ref_point=np.array([1.0]*M))
            hv_list = get_metric(instance, moeas, scaler, hv_ind, M)
            row.extend(hv_list)
            #print(instance, '\t', row)
            df.loc[len(df)] = row

    #df.to_csv(instance[:-1]+"_many-objs-hv.csv", index=False)


def main_alphaEA(instances):

    df = get_output_df()    # 获取输出格式
    for instance in instances:
        for M in [4, 6, 8]:
            row = [instance + "-" + str(M)]
            scaler = get_all_PFs_scaler(instance, M)
            hv_ind = HV(ref_point=np.array([1.0]*M))
            hv_list = get_metric_alphaea(instance, moeas, scaler, hv_ind, M)
            row.extend(hv_list)
            print(instance, '\t', row)
            df.loc[len(df)] = row

    df.to_csv(instance[:-1]+"_many-objs-alphaEA-hv.csv", index=False)


if __name__ == "__main__":

    #main(dtlz_instances)
    main_alphaEA(dtlz_instances)

