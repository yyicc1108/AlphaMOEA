import numpy as np
import pandas as pd
import geatpy as ea
from paretoset import paretoset

dtlz_instances =  ["dtlz1" , "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7"]


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


def read_label_dtlz_data(instance, run_num=30, M=4):

    n_gen = 3000

    for seed in range(run_num):

        moead_pf = pd.read_csv("../results/Part-3/"+instance[:-1]+"/moead_" + instance + "_g" + str(n_gen) + '_PF_M_' + str(M) + "_" + str(seed+1) + '.csv')
        rvea_pf = pd.read_csv("../results/Part-3/"+instance[:-1]+"/rvea_" + instance + "_g" + str(n_gen) + '_PF_M_' + str(M) + "_" + str(seed+1) + '.csv')

        PFs = np.vstack((np.array(moead_pf), np.array(rvea_pf)))

        moead_sols = pd.read_csv("../results/Part-3/solutions/moead_" + instance + "_g" + str(n_gen) + '_M_' + str(M)  + '_solutions_' + str(seed + 1) + '.csv')
        rvea_sols = pd.read_csv("../results/Part-3/solutions/rvea_" + instance + "_g" + str(n_gen) + '_M_' + str(M)  + '_solutions_' + str(seed + 1) + '.csv')

        Datas = np.vstack((np.array(moead_sols), np.array(rvea_sols)))

    mask = paretoset(PFs, sense=["min"]*M)  # 计算PF

    training_X = Datas[mask]
    training_pf = PFs[mask]
    pd.DataFrame(training_X).to_csv("../training_data/Part-3/" + instance + "_M_" + str(M) + '.csv', index=False)
    #pd.DataFrame(training_pf).to_csv("../training_data/Part-3/" + instance_name + "_M_" + str(M) + '_pf.csv', index=False)

if __name__ == '__main__':
    run_num = 30
    objs = [4, 6, 8]
    for instance_name in dtlz_instances:
        for M in objs:
            read_label_dtlz_data(instance_name, run_num, M)
