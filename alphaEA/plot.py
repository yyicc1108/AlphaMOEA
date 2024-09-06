import geatpy as ea
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import pymoo
import paretoset

from pymoo.problems import get_problem
from pymoo.util.plotting import plot
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

from sklearn.ensemble import IsolationForest

dtlz_instances = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7"]

def geatpy_1():
    problem = ea.benchmarks.DTLZ1()  # 生成问题对象
    # 构建算法
    algorithm = ea.moea_MOEAD_templet(
        problem,
        ea.Population(Encoding='RI', NIND=100),
        MAXGEN=100,  # 最大进化代数。
        logTras=1)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    # 求解
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=1,
                      outputMsg=True,
                      drawLog=True,
                      saveFlag=True,
                      dirName='result')
    print(res)

if __name__ == '__main__':

    for mop_instance in dtlz_instances:

        # ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=20)
        # pf1 = get_problem("dtlz1").pareto_front(ref_dirs)

        pf2 = pd.read_csv('../res/DTLZ/alphaEA_' + mop_instance.upper() + '.csv')

        pf2.columns = ['F1', 'F2', 'F3']

        if mop_instance == "dtlz1":
            for col in pf2.columns:
             pf2 = pf2[pf2[col] < 1]

            for col in pf2.columns:
             pf2 = pf2[pf2[col] > 0]

        # model = IsolationForest(n_estimators=100,
        #                         max_samples='auto',
        #                         contamination=float(0.05),
        #                         max_features=1.0)
        #
        # model.fit(pf2)
        # print(model.decision_function(pf2))
        # print(model.predict(pf2))
        #
        # # pf2['scores'] = model.decision_function(pf2)
        # #
        # # pf2['anomaly'] = model.predict(pf2)
        #
        # print(pf2)

        Scatter(angle=(45, 45)).add(np.array(pf2)).show()
        Scatter.save(fname=mop_instance+'.png')
    #Scatter(angle=(45, 45)).add(np.array(pf1)).show()



