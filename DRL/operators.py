
import numpy as np
import random

class evolution(object):

    def __init__(self, F1=0.001, F2=0.001, pc=1, etaC=30, lb=-100, rb=1, M=3):

        self.F1 = F1
        self.F2 = F2
        self.pc = pc
        self.etaC = etaC
        self.lb = lb
        self.rb = rb
        self.M=M

    def operator(self, pops, pf, action):

        if len(pops) > 50:
            index = random.sample(range(0, len(pops)), 50)
        else:
            index = None
            #index = random.sample(range(0, len(pops)), len(pops)//5)
        if action == 0:
            #chrPops = self.DE1(pops, index)  # rand/1
            chrPops = self.SBX(pops, index)  # SBX交叉
        elif action == 1:

            chrPops = self.DE2(pops, pf, index)  # rand/2
        else:
            chrPops = self.SBX(pops, index)  # SBX交叉

        return chrPops

    def SBX(self, pops, index=None):
        # 拷贝父代种群，以防止改变父代种群结构
        chrPops = pops.copy()

        if index==None:
            nPop = len(chrPops) if len(chrPops) % 2 == 0 else len(chrPops)-1
            for i in range(0, nPop, 2):
                chr1, chr2 = chrPops[i], chrPops[i + 1]
                if np.random.rand() < self.pc:
                    #SBX(chrPops[i], chrPops[i+1], etaC, lb, rb)  # 交叉
                    # 模拟二进制交叉
                    pos1, pos2 = np.sort(np.random.randint(0,len(chr1),2))
                    pos2 += 1
                    u = np.random.rand()
                    if u <= 0.5:
                        gamma = (2*u) ** (1/(self.etaC+1))
                    else:
                        gamma = (1/(2*(1-u))) ** (1/(self.etaC+1))
                    x1 = chr1[pos1:pos2]
                    x2 = chr2[pos1:pos2]
                    chr1[pos1:pos2], chr2[pos1:pos2] = 0.5*((1+gamma)*x1+(1-gamma)*x2), \
                        0.5*((1-gamma)*x1+(1+gamma)*x2)
                    # 检查是否符合约束
                    # chr1[chr1<self.lb] = self.lb
                    # chr1[chr1>self.rb] = self.rb
                    # chr2[chr2<self.lb] = self.lb
                    # chr2[chr2<self.rb] = self.rb

                    chrPops[i], chrPops[i + 1] = chr1, chr2
        else:
            for i in range(0, len(index), 2):
                chr1, chr2 = chrPops[index[i]], chrPops[index[i+1]]
                if np.random.rand() < self.pc:
                    #SBX(chrPops[i], chrPops[i+1], etaC, lb, rb)  # 交叉
                    # 模拟二进制交叉
                    pos1, pos2 = np.sort(np.random.randint(0,len(chr1),2))
                    pos2 += 1
                    u = np.random.rand()
                    if u <= 0.5:
                        gamma = (2*u) ** (1/(self.etaC+1))
                    else:
                        gamma = (1/(2*(1-u))) ** (1/(self.etaC+1))
                    x1 = chr1[pos1:pos2]
                    x2 = chr2[pos1:pos2]
                    chr1[pos1:pos2], chr2[pos1:pos2] = 0.5*((1+gamma)*x1+(1-gamma)*x2), \
                        0.5*((1-gamma)*x1+(1+gamma)*x2)
                    # 检查是否符合约束
                    # chr1[chr1<self.lb] = self.lb
                    # chr1[chr1>self.rb] = self.rb
                    # chr2[chr2<self.lb] = self.lb
                    # chr2[chr2<self.rb] = self.rb

                    chrPops[index[i]], chrPops[index[i+1]] = chr1, chr2

        return chrPops

    def DE1(self, pops, index=None):

        chrPops = pops.copy()
        nPop = len(chrPops)
        indices = np.arange(nPop)
        if index==None:
            for i in range(0, nPop):
                np.random.shuffle(indices)
                chrPops[i] = chrPops[i] + self.F1 * (chrPops[indices[0]] - chrPops[indices[1]])
        else:
            for i in range(0, len(index)):
                np.random.shuffle(indices)
                chrPops[index[i]] = chrPops[index[i]] + self.F1 * (chrPops[indices[0]] - chrPops[indices[1]])

        return chrPops

    def DE2(self, pops, pf, index=None):

        knee = self.point2area_distance(pf) if self.M == 3 else self.point2line_distance(pf)

        chrPops = pops.copy()
        nPop = len(chrPops)
        indices = np.arange(nPop)

        if index==None:
            for i in range(0, nPop):
                np.random.shuffle(indices)
                chrPops[i] = chrPops[knee] + self.F2 * (chrPops[indices[0]] - chrPops[indices[1]])
        else:
            for i in range(0, len(index)):
                np.random.shuffle(indices)
                chrPops[index[i]] = chrPops[knee] + self.F2 * (chrPops[indices[0]] - chrPops[indices[1]])

        return chrPops

    def define_area(self, point1, point2, point3):
        """
        法向量    ：n={A,B,C}
        空间上某点：p={x0,y0,z0}
        点法式方程：A(x-x0)+B(y-y0)+C(z-z0)=Ax+By+Cz-(Ax0+By0+Cz0)
        https://wenku.baidu.com/view/12b44129af45b307e87197e1.html
        :param point1:
        :param point2:
        :param point3:
        :param point4:
        :return:（Ax, By, Cz, D）代表：Ax + By + Cz + D = 0
        """
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        point3 = np.asarray(point3)
        AB = np.asmatrix(point2 - point1)
        AC = np.asmatrix(point3 - point1)
        N = np.cross(AB, AC)  # 向量叉乘，求法向量
        # Ax+By+Cz
        Ax = N[0, 0]
        By = N[0, 1]
        Cz = N[0, 2]
        D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])
        return Ax, By, Cz, D

    def point2area_distance(self, pf):
        """
        :param point1:数据框的行切片，三维
        :param point2:
        :param point3:
        :param point4:
        :return:点到面的距离
        """
        kneepoint = pf[np.argmin(pf, axis=0), :]
        Ax, By, Cz, D = self.define_area(kneepoint[0,:], kneepoint[1,:], kneepoint[2,:])

        index, distance = 0, 0
        for ind in range(pf.shape[0]):
            point = pf[ind, :]
            mod_d = Ax * point[0] + By * point[1] + Cz * point[2] + D
            mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
            d = abs(mod_d) / mod_area
            if d>distance:
                distance = d
                index = ind
        return index


    def point2line_distance(self, pf):

        kneepoint = pf[np.argmin(pf, axis=0), :]

        index, distance = 0, 0
        for ind in range(pf.shape[0]):
            point = pf[ind, :]
            d = self.point_distance_line(point, kneepoint[0,:], kneepoint[1,:])
            if d > distance:
                distance = d
                index = ind
        return index
    def point_distance_line(self, point, line_point1, line_point2):
        # 计算向量
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
        return distance




