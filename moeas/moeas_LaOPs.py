import time
import numpy as np
import pandas as pd
import geatpy as ea
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
import warnings
warnings.filterwarnings("ignore")

dtlz_instances = ["dtlz1",
                  "dtlz2",
    # "dtlz3",
    # "dtlz4",
                  "dtlz5", "dtlz6",
                  #"dtlz7"
                  ]
zdt_instances = ['zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6']


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

def get_df_output(g_nens):
    columns = ["Instance"]
    for _ in g_nens:
        columns.append('nsga2-' + str(_))
        columns.append('agemoea-' + str(_))
        columns.append('moead-' + str(_))
        columns.append('rvea-' + str(_))

    return pd.DataFrame(columns=columns)


def data_generator(instance, run_num, NP=100, n_gen=100, n_var=50):

    problem = get_problem(instance, n_var=n_var)

    #ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=13)
    ref_dirs = get_reference_directions("energy", problem.n_obj, NP)
    nsga2 = NSGA2(pop_size=NP)
    moead = MOEAD(ref_dirs=ref_dirs)
    rvea = RVEA(ref_dirs=ref_dirs)
    agemoea = AGEMOEA(pop_size=NP)

    nsga2_times = []
    rvea_times = []
    agemoea_times = []
    moead_times = []

    for seed in range(run_num):

        time_1 = time.time()
        res_nsga2 = minimize(problem, nsga2, ('n_gen', n_gen), seed=seed, verbose=False)
        time_2 = time.time()
        res_rvea = minimize(problem, rvea, ('n_gen', n_gen), seed=seed, verbose=False)
        time_3 = time.time()
        res_agemoea = minimize(problem, agemoea, ('n_gen', n_gen), seed=seed, verbose=False)
        time_4 = time.time()
        res_moead = minimize(problem, moead, ('n_gen', n_gen), seed=seed, verbose=False)
        time_5 = time.time()

        nsga2_times.append(time_2 - time_1)
        rvea_times.append(time_3 - time_2)
        agemoea_times.append(time_4 - time_3)
        moead_times.append(time_5 - time_4)

        #
        # if instance == "dtlz5" or instance == "dtlz6":
        #     view = (25, 10)
        # else:
        #     view = (30, 45)
        #
        # plotter = ea.PointScatter(problem_ea.M, grid=False, legend=True, title=None, view=view,
        #                           saveName='smsemoa_' + instance + '_PFs')
        # plotter.add(problem_ea.ReferObjV, color='gray', alpha=0.1, label='True PF')
        # plotter.add(np.array(res_nsga2.F), color='red', label=' PF')
        # plotter.draw()
        # plotter.show()
        #
        # plotter = ea.PointScatter(problem_ea.M, grid=False, legend=True, title=None, view=view,
        #                           saveName='smsemoa_' + instance + '_PFs')
        # plotter.add(problem_ea.ReferObjV, color='gray', alpha=0.1, label='True PF')
        # plotter.add(np.array(res_moead.F), color='red', label=' PF')
        # plotter.draw()
        # plotter.show()
        #
        # plotter = ea.PointScatter(problem_ea.M, grid=False, legend=True, title=None, view=view,
        #                           saveName='smsemoa_' + instance + '_PFs')
        # plotter.add(problem_ea.ReferObjV, color='gray', alpha=0.1, label='True PF')
        # plotter.add(np.array(res_rvea.F), color='red', label=' PF')
        # plotter.draw()
        # plotter.show()

        # Scatter().add(res_nsga2.F).show()
        # Scatter().add(res_moead.F).show()
        # Scatter().add(res_rvea.F).show()
        #
        # all_pf = np.vstack((res_nsga2.F, res_moead.F, res_rvea.F))
        # all_X = np.vstack((res_nsga2.X, res_moead.X, res_rvea.X))


        # if instance[:4] == 'dtlz':
        #     pd.DataFrame(res_nsga2.F).to_csv(
        #         "../results/Part-5/dtlz/nsga2_" + instance + "_g" + str(n_gen) + '_PF_' + str(seed + 1) + '.csv', index=False)
        #     pd.DataFrame(res_moead.F).to_csv(
        #         "../results/Part-5/dtlz/moead_" + instance + "_g" + str(n_gen) + '_PF_' + str(seed + 1) + '.csv', index=False)
        #     pd.DataFrame(res_rvea.F).to_csv(
        #         "../results/Part-5/dtlz/rvea_" + instance + "_g" + str(n_gen) + '_PF_' + str(seed + 1) + '.csv',
        #         index=False)
        # else:
        #     pd.DataFrame(res_nsga2.F).to_csv(
        #         "../results/Part-5/zdt/nsga2_" + instance + "_g" + str(n_gen) + '_PF_' + str(seed + 1) + '.csv', index=False)
        #     pd.DataFrame(res_moead.F).to_csv(
        #         "../results/Part-5/zdt/moead_" + instance + "_g" + str(n_gen) + '_PF_' + str(seed + 1) + '.csv', index=False)
        #     pd.DataFrame(res_rvea.F).to_csv(
        #         "../results/Part-5/zdt/rvea_" + instance + "_g" + str(n_gen) + '_PF_' + str(seed + 1) + '.csv', index=False)
        #
        #
        # pd.DataFrame(res_nsga2.X).to_csv(
        #     "../results/Part-5/solutions/nsga2_" + instance + "_g" + str(n_gen) + '_solutions_' + str(seed + 1) + '.csv',
        #     index=False)
        # pd.DataFrame(res_moead.X).to_csv(
        #     "../results/Part-5/solutions/moead_" + instance + "_g" + str(n_gen) + '_solutions_' + str(seed + 1) + '.csv',
        #     index=False)
        # pd.DataFrame(res_rvea.X).to_csv(
        #     "../results/Part-5/solutions/rvea_" + instance + "_g" + str(n_gen) + '_solutions_' + str(seed + 1) + '.csv',
        #     index=False)

    return (np.mean(nsga2_times), np.mean(agemoea_times), np.mean(moead_times), np.mean(rvea_times))


if __name__ == '__main__':

    run_num = 10
    #g_nens = [2000]

    # df_zdt = get_df_output(g_nens)
    # for instance in zdt_instances:
    #     s_time = time.time()
    #     for n_gen in g_nens:
    #         row = [instance]
    #         times = data_generator(instance, run_num, NP=100, n_gen=n_gen)
    #         row.extend([times[0], times[1], times[2], times[3]])
    #         print(row)
    #     df_zdt.loc[len(df_zdt)] = row
    #     print("Instance: {0}, \t\t Running times: {1}s ".format(instance, time.time() - s_time))
    #
    # print(df_zdt)
    # df_zdt.to_csv("ZDT-MOEAs-Eva-Running-Times.csv", index=False)

    g_nens = [3000]
    D = [50, 100, 200, 300]
    df_dtlz = get_df_output(g_nens)
    for instance in dtlz_instances:
        for n_var in D:
            s_time = time.time()
            row = [instance+"-d-"+str(n_var)]
            for n_gen in g_nens:
                times = data_generator(instance, run_num, NP=100, n_gen=n_gen, n_var=n_var)
                row.extend([times[0], times[1], times[2], times[3]])
                print(row)
            df_dtlz.loc[len(df_dtlz)] = row
            print("Instance: {0},  \t\t D: {1}, \t\t Running times: {2}s ".format(instance, n_var, round(time.time() - s_time, 4)))

    print(df_dtlz)
    df_dtlz.to_csv("DTLZ-MOEAs-LaOPs-Running-Times.csv", index=False)

