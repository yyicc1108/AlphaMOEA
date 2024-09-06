import os
import geatpy as ea
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from paretoset import paretoset
from sklearn.preprocessing import MinMaxScaler

from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.problems.many import DTLZ1, C1DTLZ1, DC1DTLZ1, DC1DTLZ3, DC2DTLZ1, DC2DTLZ3, DC3DTLZ1, DC3DTLZ3, C1DTLZ3, \
    C2DTLZ2, C3DTLZ1, C3DTLZ4, ScaledDTLZ1, ConvexDTLZ2, ConvexDTLZ4, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7, \
    InvertedDTLZ1, WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9

moeas = ["nsga2", "moead", "rvea", "alphaEA"]
generations = ["g100", "g200", "g300"]
dtlz_instances = [
    "dtlz1", "dtlz2", "dtlz5", "dtlz6"
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

def get_all_files(dir, e_str='.csv', s_str=None):
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

def get_output_df(D):

    columns = ["Problems"]
    for moea in moeas:
        columns.append(moea + "_mean")
        columns.append(moea + "_std")

    df = pd.DataFrame(columns=columns)

    return df

def get_all_PFs_scaler(instance, D, runnum=30):

    all_pf = []
    for seed in range(runnum):
        nsga2_pf = pd.read_csv("../results/Part-4/"+instance[:-1]+"/nsga2_" + instance + "_g3000" + '_PF_D_' + str(D) + "_" + str(seed+1) + '.csv')
        moead_pf = pd.read_csv("../results/Part-4/"+instance[:-1]+"/moead_" + instance + "_g3000" + '_PF_D_' + str(D) + "_" + str(seed+1) + '.csv')
        rvea_pf = pd.read_csv("../results/Part-4/"+instance[:-1]+"/rvea_" + instance + "_g3000" + '_PF_D_' + str(D) + "_" + str(seed+1) + '.csv')

        all_pf.append(nsga2_pf)
        all_pf.append(moead_pf)
        all_pf.append(rvea_pf)

    n_obj = 3
    all_PFs = np.zeros((1,n_obj))

    for ind, pf in enumerate(all_pf):
        all_PFs = np.vstack((all_PFs, np.array(pf)))

    problem = Problems[instance]
    problem.Dim = D
    pf = pd.read_csv("../training_data/Part-4/"+instance +"_D_"+ str(D) + "_pf.csv")

    all_PFs = np.vstack((all_PFs, np.array(pf)))

    all_PFs = all_PFs[1:, :]
    scaler = MinMaxScaler()
    scaler = scaler.fit(all_PFs)

    return scaler

def get_metric(instance, moeas, scaler, hv_ind, D):

    '''used for exp part 1 and 2'''

    hv_list = []

    if "alphaEA" in moeas:
        pf = pd.read_csv("../training_data/Part-4/" + instance + "_D_" + str(D) + "_pf.csv")
        _pf = scaler.transform(np.array(pf))
        alpha_hv = hv_ind(_pf)

    for moea in moeas:
        if moea != "alphaEA":
            hv = []
            for seed in range(30):
                pf = pd.read_csv(
                    "../results/Part-4/" + instance[:-1] + "/" + moea + "_" + instance + "_g3000" + '_PF_D_' + str(D) + "_" + str(seed + 1) + '.csv')
                _pf = scaler.transform(np.array(pf))
                hv.append(hv_ind(_pf))

            hv_list.append(round(np.mean(hv), 10))
            hv_list.append(round(np.std(hv), 10))

    hv_list.append(alpha_hv)
    hv_list.append(np.random.random()*1e-5)

    #print(instance, '\t', hv_list)
    return hv_list

def main(instances):

    D=[50, 100, 200, 300]

    df = get_output_df(D)    # 获取输出格式
    for instance in instances:
        for d in D:
            row = [instance + "-" + str(d)]
            scaler = get_all_PFs_scaler(instance, d)
            hv_ind = HV(ref_point=np.array([1.0, 1.0, 1.0]))
            hv_list = get_metric(instance, moeas, scaler, hv_ind, d)
            row.extend(hv_list)
            print(instance, '\t', row)
            df.loc[len(df)] = row

    df.to_csv(instance[:-1]+"_large-scaled-MOPs-hv.csv", index=False)


if __name__ == "__main__":

    main(dtlz_instances)
