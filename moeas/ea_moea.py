import time
import numpy as np
import pandas as pd
import geatpy as ea
from geatpy import moea_MOEAD_templet, moea_NSGA2_templet, moea_RVEA_templet, moea_NSGA3_templet

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


dtlz_instances = ['dtlz1']
zdt_instances = ['zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6']

def get_df_output(n_gens):
    columns = ["Instance"]
    for n_gen in n_gens:
        columns.append('nsga2' + str(n_gen))
        #columns.append('nsga3' + str(n_gen))
        columns.append('moead' + str(n_gen))
        columns.append('rvea' + str(n_gen))

    return pd.DataFrame(columns=columns)


def test_moesd(instance, n_gen, run_num):

    nsga2_times, nsga3_times, moead_times, rvea_times = [], [], [], []

    for seed in range(run_num):
        problem = Problems[instance]

        nsga2 = moea_NSGA2_templet(problem, ea.Population(Encoding='RI', NIND=100), MAXGEN=n_gen, drawing=0)
        #nsga3 = moea_NSGA3_templet(problem, ea.Population(Encoding='RI', NIND=100), MAXGEN=n_gen, drawing=0)
        moead = moea_MOEAD_templet(problem, ea.Population(Encoding='RI', NIND=100), MAXGEN=n_gen, drawing=0)
        rvea = moea_RVEA_templet(problem, ea.Population(Encoding='RI', NIND=100), MAXGEN=n_gen , drawing=0)

        # 求解
        nsga2_res = ea.optimize(nsga2, logTras=0, drawing=0, verbose=False, outputMsg=False, drawLog=False)
        #nsga3_res = ea.optimize(nsga3, logTras=0, drawing=0, verbose=False, outputMsg=False, drawLog=False)
        moead_res = ea.optimize(moead, logTras=0, drawing=0, verbose=False, outputMsg=False, drawLog=False)
        rvea_res = ea.optimize(rvea, logTras=0, drawing=0, verbose=False, outputMsg=False, drawLog=False)

        nsga2_times.append(nsga2_res['executeTime'])
        #nsga3_times.append(nsga3_res['executeTime'])
        moead_times.append(moead_res['executeTime'])
        rvea_times.append(rvea_res['executeTime'])

        if instance[:4] == 'dtlz':
            pd.DataFrame(nsga2_res['ObjV']).to_csv("../results/ea/dtlz/nsga2_" + instance + "_g" + str(n_gen) + '_PF_' + str(seed + 1) + '.csv', index=False)
            pd.DataFrame(moead_res['ObjV']).to_csv("../results/ea/dtlz/moead_" + instance + "_g" + str(n_gen) + '_PF_' + str(seed + 1) + '.csv', index=False)
            pd.DataFrame(rvea_res['ObjV']).to_csv("../results/ea/dtlz/rvea_" + instance + "_g" + str(n_gen) + '_PF_' + str(seed + 1) + '.csv', index=False)
        else:
            pd.DataFrame(nsga2_res['ObjV']).to_csv("../results/ea/zdt/nsga2_" + instance + "_g" + str(n_gen) + '_PF_' + str(seed + 1) + '.csv', index=False)
            pd.DataFrame(moead_res['ObjV']).to_csv("../results/ea/zdt/moead_" + instance + "_g" + str(n_gen) + '_PF_' + str(seed + 1) + '.csv', index=False)
            pd.DataFrame(rvea_res['ObjV']).to_csv("../results/ea/zdt/rvea_" + instance + "_g" + str(n_gen) + '_PF_' + str(seed + 1) + '.csv', index=False)

        pd.DataFrame(nsga2_res['optPop'].Chrom).to_csv("../results/ea/solutions/nsga2_" + instance + "_g" + str(n_gen) + '_solutions_' + str(seed + 1) + '.csv', index=False)
        pd.DataFrame(moead_res['optPop'].Chrom).to_csv("../results/ea/solutions/moead_" + instance + "_g" + str(n_gen) + '_solutions_' + str(seed + 1) + '.csv', index=False)
        pd.DataFrame(rvea_res['optPop'].Chrom).to_csv("../results/ea/solutions/rvea_" + instance + "_g" + str(n_gen) + '_solutions_' + str(seed + 1) + '.csv', index=False)

    return (np.mean(nsga2_times), np.mean(moead_times), np.mean(rvea_times))


if __name__ == '__main__':

    run_num = 30
    n_gens = [500]

    df_dtlz = get_df_output(n_gens)
    df_zdt = get_df_output(n_gens)

    for instance in dtlz_instances:
        row = [instance]
        for n_gen in n_gens:
            s_time = time.time()
            times = test_moesd(instance, n_gen, run_num)
            row.extend([times[0], times[1], times[2]])

        df_dtlz.loc[len(df_dtlz)] = row

        print("Instance: {0}, \t\t Running times: {1}s ".format(instance, time.time() - s_time))

    print(df_dtlz)
    df_dtlz.to_csv("../results/times/DTLZ-ea-MOEAs-running-times-1.csv", index=False)

    # for instance in zdt_instances:
    #     row = [instance]
    #     for n_gen in n_gens:
    #         s_time = time.time()
    #         times = test_moesd(instance, n_gen, run_num)
    #         row.extend([times[0], times[1]])
    #
    #     df_zdt.loc[len(df_zdt)] = row
    #
    #     print("Instance: {0}, \t\t Running times: {1}s ".format(instance, time.time() - s_time))
    #
    # print(df_zdt)
    # df_zdt.to_csv("../results/times/ZDT-ea-MOEAs-running-times.csv", index=False)

