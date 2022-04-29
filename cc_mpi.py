# -*-coding:utf-8-*-
import copy
import itertools
import os
import pickle
from time import time

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from pandas import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

Comm = MPI.COMM_WORLD
Rank = Comm.Get_rank() - 1
Size = Comm.Get_size() - 1


class SlaverBean:
    def __init__(self, slaver):
        self.position = slaver.position  # 坐标
        self.cityNum = slaver.cityNum  # 城市数量，染色体长度
        self.dist = slaver.dist  # 距离矩阵
        self.bestDistance = slaver.bestDistance  # 最佳长度
        self.bestPath = slaver.bestPath  # 最佳路径
        self.oldPopulation = slaver.oldPopulation  # 父代种群
        self.newPopulation = slaver.newPopulation  # 子代种群
        self.fitness = slaver.fitness  # 个体的适应度
        self.Pi = slaver.Pi  # 个体的累积概率


class GA:
    def __init__(self, m, t, p_cross, p_mutate, group_count, migrate_rate, wait, input_file, output_file):
        self.inputFile = input_file  # 数据源
        self.outputFile = output_file  # 数据源
        self.position = []  # 坐标
        self.M = m  # 种群规模
        self.T = t  # 运行代数
        self.t = 0  # 当前代数
        self.p_cross = p_cross  # 交叉概率
        self.p_mutate = p_mutate  # 变异概率
        self.cityNum = 0  # 城市数量，染色体长度
        self.dist = []  # 距离矩阵
        self.bestDistance = np.inf  # 最佳长度
        self.bestPath = []  # 最佳路径
        self.oldPopulation = []  # 父代种群
        self.newPopulation = []  # 子代种群
        self.fitness = []  # 个体的适应度
        self.Pi = []  # 个体的累积概率
        self.record = []  # 记录适应度变化
        self.groupCount = group_count  # 组数
        self.tribeCount = Size // group_count  # 组内种群书
        self.migrateRate = migrate_rate  # 每个migrateRate代组间迁移一次
        self.isFinish = False  # 该种群是否不在变化
        self.wait = wait

    def copy(self, slaver: SlaverBean):
        self.position = slaver.position  # 坐标
        self.cityNum = slaver.cityNum  # 城市数量，染色体长度
        self.dist = slaver.dist  # 距离矩阵
        self.bestDistance = slaver.bestDistance  # 最佳长度
        self.bestPath = slaver.bestPath  # 最佳路径
        self.oldPopulation = slaver.oldPopulation  # 父代种群
        self.newPopulation = slaver.newPopulation  # 子代种群
        self.fitness = slaver.fitness  # 个体的适应度
        self.Pi = slaver.Pi  # 个体的累积概率
        self.M = len(self.oldPopulation)

    # 读取文件
    def read_file(self, filepath):
        with open(filepath) as infile:
            for line in infile:
                split = line.strip().split()
                self.position.append([float(split[1]), float(split[2])])

    # 初始化dist矩阵
    def init_dist(self):
        self.cityNum = len(self.position)  # 城市数量，染色体长度
        self.dist = np.zeros([self.cityNum, self.cityNum], dtype=int)
        for i in range(self.cityNum):
            for j in range(i, self.cityNum):
                self.dist[i][j] = self.dist[j][i] = self.distance(self.position[i], self.position[j])

    # 随机初始化种群
    def init_population(self):
        # 随机初始化种群
        for k in range(self.M):
            tmp = np.arange(self.cityNum)
            np.random.shuffle(tmp)
            self.oldPopulation.append(tmp)

    # 更新个体适应度并做好记录
    def update_fitness(self):
        self.fitness.clear()
        for i in range(self.M):
            self.fitness.append(self.evaluate(self.oldPopulation[i]))
        self.record.append(np.sum(self.fitness) / self.M)

    # 计算某个染色体的实际距离作为染色体适应度
    def evaluate(self, chromosome):
        l = 0
        for i in range(1, self.cityNum):
            l += self.dist[chromosome[i - 1]][chromosome[i]]
        l += self.dist[chromosome[0]][chromosome[self.cityNum - 1]]  # 回到起点
        return l

    # 计算欧氏距离矩阵
    @staticmethod
    def distance(pos1, pos2):
        return np.around(np.sqrt(np.sum(np.power(np.array(pos1) - np.array(pos2), 2))))

    # 适应度转化函数
    @staticmethod
    def fit_func(fit):
        return 10000 / fit

    # 计算种群中每个个体的累积概率
    def count_rate(self):
        tmp_fit = self.fit_func(np.array(self.fitness))
        fit_sum = np.sum(tmp_fit)
        self.Pi = tmp_fit / fit_sum
        self.Pi = list(itertools.accumulate(self.Pi))
        self.Pi[self.M - 1] = np.round(self.Pi[self.M - 1])  # 最后四舍五入保证累计概率的最后一个值为1

    # 轮盘挑选子代个体
    def select_child(self):
        self.newPopulation.clear()
        for i in range(0, self.M):
            rate = np.random.random(1)
            for oldId in range(self.M):
                if self.Pi[oldId] >= rate:
                    self.newPopulation.append(copy.deepcopy(self.oldPopulation[oldId]))
                    break
        self.oldPopulation.clear()

    # 进化种群
    def evolution(self):
        self.select_child()  # 选择
        rand = np.arange(self.M)
        np.random.shuffle(rand)
        for k in range(1, self.M, 2):
            rate_c = np.random.random(1)
            if rate_c < self.p_cross:  # 交叉
                self.order_cross(rand[k], rand[k - 1])
            rate_m = np.random.random(1)
            if rate_m < self.p_mutate:  # 变异
                self.variation(rand[k])
            rate_m = np.random.random(1)
            if rate_m < self.p_mutate:
                self.variation(rand[k - 1])

    # 产生2个索引，用于交叉和变异
    def random_range(self):
        left = 0
        right = 0
        while left == right:
            left = np.random.randint(0, self.cityNum)
            right = np.random.randint(0, self.cityNum)
        ran = np.sort([left, right])
        left = ran[0]
        right = ran[1]
        return left, right

    # 变异算子，翻转一段基因
    def variation(self, k):
        ran = self.random_range()
        left = ran[0]
        right = ran[1]
        while left < right:
            tmp = self.newPopulation[k][left]
            self.newPopulation[k][left] = self.newPopulation[k][right]
            self.newPopulation[k][right] = tmp
            left = left + 1
            right = right - 1

    # 映射交叉算子(互换片段，映射去重)
    def order_cross(self, k1, k2):
        ran = self.random_range()
        left = ran[0]
        right = ran[1]
        map1 = {}
        map2 = {}
        old1 = copy.deepcopy(self.newPopulation[k1])
        old2 = copy.deepcopy(self.newPopulation[k2])
        for i in range(left, right + 1):
            map1[self.newPopulation[k1][i]] = self.newPopulation[k2][i]
            map2[self.newPopulation[k2][i]] = self.newPopulation[k1][i]

        for i in range(self.cityNum):
            g = self.newPopulation[k1][i]
            if i < left or i > right:
                while map2.get(g) is not None:  # 非交换部分，由于有些交换可以抵消，所以要循环映射直到找不到下一个映射
                    g = map2[g]
                self.newPopulation[k1][i] = g
            else:  # 交换的部分直接映射
                self.newPopulation[k1][i] = map1[g]

        for i in range(self.cityNum):
            g = self.newPopulation[k2][i]
            if i < left or i > right:
                while map1.get(g) is not None:
                    g = map1[g]
                self.newPopulation[k2][i] = g
            else:
                self.newPopulation[k2][i] = map2[g]

        # 只有比父母好才能替代父母
        if self.evaluate(old1) < self.evaluate(self.newPopulation[k2]):
            self.newPopulation[k2] = old1
        if self.evaluate(old2) < self.evaluate(self.newPopulation[k1]):
            self.newPopulation[k1] = old2

    # 获取最好的个体
    def get_best(self):
        best_id = np.argmin(self.fitness)
        best_distance = self.fitness[best_id]
        if best_distance < self.bestDistance:
            self.bestDistance = best_distance
            self.bestPath = copy.deepcopy(self.oldPopulation[best_id])

    def init_master(self):
        self.cityNum = len(self.position)
        self.record.append(0)
        self.w = self.wait

    def inner_migrate(self):
        # 把老一代最好的放入现在刚成为老一代的新一代
        group = Rank // self.tribeCount * self.tribeCount
        dest = (Rank + 1) % self.tribeCount + group
        src = (Rank - 1 + self.tribeCount) % self.tribeCount + group
        Comm.send(self.bestPath, dest=dest + 1)
        self.oldPopulation[0] = Comm.recv(source=src + 1)

    def outer_migrate(self):
        if Rank % self.tribeCount == 0:
            dest = (Rank + self.tribeCount) % Size
            src = (Rank - self.tribeCount + Size) % Size
            Comm.send(self.oldPopulation, dest=dest + 1)
            self.oldPopulation = Comm.recv(source=src + 1)

    def gather_fitness(self):
        records = Comm.gather(self.record[self.t], root=0)  # 实际上只有0号进程获取了，对于其他进程record是None，不能直接求和除Size
        if Rank == -1:
            self.record[self.t] = np.sum(records) / Size

    def dump(self):
        parent = self.checkpointDir + str(self.groupRank) + '-' + str(self.rank) + '/'
        n = self.t // self.migrateRate
        if n == 0:
            return
        if not os.path.isdir(parent):
            os.mkdir(parent)
        with open(parent + str(n) + '.slaver', 'wb')as f:
            pickle.dump(SlaverBean(self), f)
        if n > self.maxCheckpoint:
            os.remove(parent + str(n - self.maxCheckpoint) + '.slaver')

    def load(self):
        parent = self.checkpointDir + str(self.groupRank) + '-' + str(self.rank) + '/'
        if os.path.isdir(parent):
            files = os.listdir(parent)
            if len(files) == 0:
                return False
            last_point = files[-1]
            with open(parent + last_point, 'rb')as f:
                self.copy(pickle.loads(f.read()))
            return True
        return False

    # 开始GA
    def run(self):
        if Rank == -1:
            self.read_file(self.inputFile)
        self.position = Comm.bcast(self.position, root=0)  # 广播坐标集
        if Rank != -1:
            self.init_dist()  # 初始化距离矩阵
            self.init_population()  # 初始化种群
            self.update_fitness()  # 初始化适应度
            self.count_rate()  # 初始化累计概率
            self.get_best()  # 得到最好的个体
        else:
            self.init_master()

        self.gather_fitness()

        while self.t < self.T:
            self.t += 1
            self.isFinish = False
            if Rank != -1:
                self.evolution()
                self.oldPopulation = copy.deepcopy(self.newPopulation)
                # 组间交换
                if self.t % self.migrateRate == 0 and self.groupCount > 1:
                    self.outer_migrate()
                # 组内交换
                elif self.tribeCount > 1:
                    self.inner_migrate()
                self.update_fitness()
                self.count_rate()
                self.get_best()
            else:
                self.record.append(0)

            self.gather_fitness()

            # 组间迁移后master收集最优结果
            if self.t % self.migrateRate == 0:
                best_paths = Comm.gather(self.bestPath, root=0)
                best_distances = Comm.gather(self.bestDistance, root=0)
                if Rank == -1:
                    best_id = np.argmin(best_distances)
                    bestPath = best_paths[best_id]
                    bestDistance = best_distances[best_id]
                    if bestDistance < self.bestDistance:
                        self.bestDistance = bestDistance
                        self.bestPath = bestPath
                        self.w = self.wait
                    else:
                        self.w -= 1
                    # 0号判断是否完成
                    if self.w == 0:
                        self.isFinish = True
                    else:
                        self.isFinish = False
                    # 检查是否加入更新
                    self.check_add()
            # 广播让其他进程也结束
            self.isFinish = Comm.bcast(self.isFinish, root=0)
            if self.isFinish:
                break

        # 搜集其他种群最好的个体
        if Rank == -1:
            print("结果:")
            print("终止代数:" + str(self.t))
            print("最优路径:" + str(self.bestPath))
            print("最优距离:" + str(self.bestDistance))

    # 显示结果
    def show(self):
        plt.title('TSP-GA')
        ax1 = plt.subplot(221)
        ax1.set_title('original')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        for point in self.position:
            plt.plot(point[0], point[1], marker='o', color='k')

        ax2 = plt.subplot(222)
        ax2.set_title('fitness tendency')
        ax2.set_xlabel('t')
        ax2.set_ylabel('mean loss')
        for i in range(1, len(self.record)):
            plt.plot([i, i - 1], [self.record[i], self.record[i - 1]], marker='o', color='k', markersize='1')

        ax3 = plt.subplot(223)
        ax3.set_title('way')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        for point in self.position:
            plt.plot(point[0], point[1], marker='o', color='k')
        for i in range(1, self.cityNum):
            plt.plot([self.position[self.bestPath[i]][0], self.position[self.bestPath[i - 1]][0]],
                     [self.position[self.bestPath[i]][1], self.position[self.bestPath[i - 1]][1]], color='g')
        plt.plot([self.position[self.bestPath[0]][0], self.position[self.bestPath[self.cityNum - 1]][0]],
                 [self.position[self.bestPath[0]][1], self.position[self.bestPath[self.cityNum - 1]][1]], color='g')

        plt.savefig(self.outputFile + '-' + str(self.bestDistance) + '.jpg')
        plt.show()


if __name__ == '__main__':
    t1 = time()
    ga = GA(m=800 // Size,
            t=10000,
            p_cross=0.7,
            p_mutate=0.05,
            group_count=2,
            migrate_rate=100,
            wait=10,
            input_file="./data/st70.txt",
            output_file="./result/st70.txt")
    ga.run()
    t2 = time()
    if Rank == -1:
        print("耗时:" + str(t2 - t1))
        ga.show()
