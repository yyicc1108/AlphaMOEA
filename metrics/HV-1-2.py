import os
import geatpy as ea
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from paretoset import paretoset
from sklearn.preprocessing import MinMaxScaler

from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

moeas = [
    "nsga2",
    "agemoea",
    "moead",
    "rvea",
    "alphaEA"
]
generations = ["g100", "g200", "g300"]
zdt_instances = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"]
dtlz_instances = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7"]

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

def get_output_df():

    columns = ["Problems"]
    for moea in moeas:
        columns.append(moea + "_mean")
        columns.append(moea + "_std")
        # if moea != "alphaEA":
        #     columns.append("Sig.")
        #columns.append("Time")

    df = pd.DataFrame(columns=columns)

    return df

def get_all_PFs_scaler(instance):

    all_files = []
    for moea in moeas:
        all_files.append(get_all_files('../results/Part-1-2/' + instance[:-1], s_str=moea + '_' + instance, e_str='.csv'))

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

def get_metric(instance, moeas, scaler, hv_ind):

    '''used for exp part 1 and 2'''

    hv_list = []

    if "alphaEA" in moeas:
        name = moeas[moeas.index("alphaEA")]
        PF_files = get_all_files('../results/Part-1-2/' + instance[:-1], s_str=name + '_' + instance, e_str='csv')
        alpha_hv_list = []
        for file in PF_files:
            pf = pd.read_csv(file, header=0)
            _pf = scaler.transform(np.array(pf))
            alpha_hv_list.append(hv_ind(_pf))

    for moea in moeas:
        if moea != "alphaEA":
            PF_files = get_all_files('../results/Part-1-2/' + instance[:-1], s_str=moea + '_' + instance, e_str='csv')
            hv = []
            for file in PF_files:
                pf = pd.read_csv(file, header=0)
                #print(hv_ind(np.array(pf)))
                _pf = scaler.transform(np.array(pf))
                hv.append(hv_ind(_pf))

            # stat, p = wilcoxon(hv, alpha_hv_list)
            # sig = '+/-' if p < 0.05 else '='

            hv_list.append(round(np.mean(hv), 10))
            hv_list.append(round(np.std(hv), 10))
            #hv_list.append(sig)

    hv_list.append(round(np.mean(alpha_hv_list), 10))
    hv_list.append(round(np.std(alpha_hv_list), 10))

    #print(instance, '\t', hv_list)
    return hv_list

def main(instances):

    df = get_output_df()    # 获取输出格式
    moea_times = pd.read_csv('../results/Part-1-2/times/' + instances[0][:-1].upper()+"-MOEAs-running-times.csv", header=0)
    alpha_times = pd.read_csv('../results/Part-1-2/times/' + instances[0][:-1].upper()+"-AlphaEA-running-times.csv", header=0)
    for index, instance in enumerate(instances):
        row = [instance]
        scaler = get_all_PFs_scaler(instance)
        hv_ind = HV(ref_point=np.array([1.0, 1.0, 1.0])) if instance[0] == 'd' else HV(ref_point=np.array([1.0, 1.0]))
        hv_list = get_metric(instance, moeas, scaler, hv_ind)
        # hv_list.insert(3, round(moea_times.iloc[index, 1], 2))
        # hv_list.insert(7, round(moea_times.iloc[index, 2], 2))
        # hv_list.insert(11, round(moea_times.iloc[index, 3], 2))
        # hv_list.insert(13, round(moea_times.iloc[index, 3], 2))
        # hv_list.append(round(alpha_times.iloc[index, 1], 2))
        row.extend(hv_list)
        print(instance, '\t', row)
        df.loc[len(df)] = row

    save_name = instance[:-1]+"_hv-1.csv" if instance[0] == 'z' else instance[:-1]+"_hv-2.csv"
    df.to_csv(save_name, index=False)


if __name__ == "__main__":
    main(zdt_instances)
    main(dtlz_instances)

