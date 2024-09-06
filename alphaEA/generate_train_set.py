import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from paretoset import paretoset


instances = ['dtlz1','dtlz2','dtlz3','dtlz4','dtlz5','dtlz6','dtlz7','zdt1','zdt2','zdt3','zdt4','zdt6']


def data_generator(instance_name, NP=100, n_gen=100):

    problem = get_problem(instance_name)
    ref_dirs = get_reference_directions("energy", problem.n_obj, NP, seed=1)
    nsga2 = NSGA2(pop_size=NP)
    moead = MOEAD(ref_dirs, n_neighbors=10, prob_neighbor_mating=0.7)
    smsemoa = SMSEMOA(pop_size=NP)

    res_nsga2 = minimize(problem, nsga2, ('n_gen', n_gen), seed=1, verbose=False)
    res_moead = minimize(problem, moead, ('n_gen', n_gen), seed=1, verbose=False)
    res_smsemoa = minimize(problem, smsemoa, ('n_gen', n_gen), seed=1, verbose=False)

    all_pf = np.vstack((res_nsga2.F, res_moead.F, res_smsemoa.F))
    all_X = np.vstack((res_nsga2.X, res_moead.X, res_smsemoa.X))

    if instance_name[:4] == 'dtlz':
        mask = paretoset(all_pf, sense=["min", "min",  "min"])  # 计算PF
    else:
        mask = paretoset(all_pf, sense=["min", "min"])

    training_X = all_X[mask]
    #PF = all_pf[mask]
    pd.DataFrame(training_X).to_csv("../training_data/" + instance_name + "_dim_" + str(problem.n_var) + '.csv', index=False)

def read_label_data(instance, run_num):

    problem = get_problem(instance, n_var=10) if instance[:4] == 'dtlz' else get_problem(instance, n_var=30)

    n_gen = 500 if instance[:4] == 'dtlz' else 300


    for seed in range(run_num):

        nsga2_pf = pd.read_csv("../results/"+instance[:-1]+"/nsga2_" + instance + "_g" + str(n_gen) + '_PF_' + str(seed+1) + '.csv', index=False)
        moead_pf = pd.read_csv("../results/"+instance[:-1]+"/moead_" + instance + "_g" + str(n_gen) + '_PF_' + str(seed+1) + '.csv', index=False)
        smsemoa_pf = pd.read_csv("../results/"+instance[:-1]+"/smsemoa_" + instance + "_g" + str(n_gen) + '_PF_' + str(seed+1) + '.csv', index=False)

        all_pf = np.vstack((np.array(nsga2_pf), np.array(moead_pf), np.array(smsemoa_pf)))

        nsga2_sols = pd.read_csv("../results/solutions/nsga2_" + instance + "_g" + str(n_gen) + '_solutions_' + str(seed + 1) + '.csv', index=False)
        moead_sols = pd.read_csv("../results/solutions/moead_" + instance + "_g" + str(n_gen) + '_solutions_' + str(seed + 1) + '.csv', index=False)
        smsemoa_sols = pd.read_csv("../results/solutions/smsemoa_" + instance + "_g" + str(n_gen) + '_solutions_' + str(seed + 1) + '.csv', index=False)

        all_X = np.vstack((np.array(nsga2_sols), np.array(moead_sols), np.array(smsemoa_sols)))

        if instance_name[:4] == 'dtlz':
            mask = paretoset(all_pf, sense=["min", "min", "min"])  # 计算PF
        else:
            mask = paretoset(all_pf, sense=["min", "min"])

        training_X = all_X[mask]
        # PF = all_pf[mask]
        pd.DataFrame(training_X).to_csv("../training_data/" + instance_name + "_dim_" + str(problem.n_var) + '.csv', index=False)


if __name__ == '__main__':
    run_num = 30
    for instance_name in instances:
        read_label_data(instance_name, run_num)
