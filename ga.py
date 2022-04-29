# -*-coding:utf-8-*-
import copy
import itertools
import os
import pickle
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from pandas import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


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
        self.t = slaver.t


class Slaver:
    def __init__(self, M, p_cross, p_mutate, max_checkpoint):
        self.position = []  # 坐标
        self.M = M  # 种群规模
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
        self.migrateRate = None  # 每个migrateRate代组间迁移一次
        self.isFinish = False  # 该种群是否不在变化
        self.outer_comm = MPI.Comm.Get_parent()  # 外部通讯
        self.inner_comm = MPI.COMM_WORLD  # 内部通讯
        self.rank = self.inner_comm.Get_rank()  # 内部序号
        self.size = self.inner_comm.Get_size()  # 内部大小
        self.groupSize = 0  # 全局组数
        self.groupRank = -1  # 全局组号
        self.maxCheckpoint = max_checkpoint
        self.checkpointDir = 'checkpoint/'

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
        self.t = slaver.t
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
        for i in range(self.M):
            rate = np.random.random(1)
            for oldId in range(len(self.Pi)):
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

    def inner_migrate(self):
        # 把老一代最好的放入现在刚成为老一代的新一代
        dest = (self.rank + 1) % self.size
        src = (self.rank - 1 + self.size) % self.size
        self.inner_comm.send(self.bestPath, dest=dest)
        self.oldPopulation[0] = self.inner_comm.recv(source=src)

    def outer_migrate(self):
        if self.rank == 0:
            self.outer_comm.send(self.oldPopulation, dest=0)
            self.oldPopulation = self.outer_comm.recv(source=0)
            self.M = len(self.oldPopulation)

    def synchro(self):
        # 搜集最优
        self.outer_comm.gather((self.bestDistance, self.bestPath), root=0)
        # 获取全局状态
        self.groupSize, self.isFinish = self.outer_comm.bcast(None, root=0)
        # 负载均衡
        add_size, all_size = self.outer_comm.bcast(None, root=0)
        if add_size > 0:
            drop_size = add_size * self.M // all_size
            self.outer_comm.gather(self.oldPopulation[:drop_size], root=0)
            self.oldPopulation = self.oldPopulation[drop_size:]
            self.M = len(self.oldPopulation)

        print(str(self.groupSize) + '-' + str(self.rank) + '种群数量:' + str(self.M))

    def slaver_run(self):
        self.groupRank, isInit = self.outer_comm.bcast(None, root=0)
        if isInit:
            if not self.load():
                self.outer_comm.gather(False, root=0)
                self.position = self.outer_comm.recv(None, source=0)  # 广播坐标集
                self.init_dist()  # 初始化距离矩阵
                self.init_population()  # 初始化种群
            else:
                self.outer_comm.gather(True, root=0)
        else:
            self.position = self.outer_comm.recv(None, source=0)
            self.oldPopulation = self.outer_comm.recv(None, source=0)
            self.M = len(self.oldPopulation)
            self.init_dist()  # 初始化距离矩阵
        self.groupSize, self.isFinish, self.migrateRate = self.outer_comm.bcast(None, root=0)
        while True:
            self.t += 1  # t自增必须方开头，不然有结点和master不同步，差1
            self.update_fitness()
            self.count_rate()
            self.get_best()
            self.outer_comm.gather(self.record[-1], root=0)

            self.evolution()
            self.oldPopulation = copy.deepcopy(self.newPopulation)
            # 组间交换
            if self.t != 0 and self.t % self.migrateRate == 0:
                # 组间迁移后master收集最优结果
                if self.groupSize > 1:
                    self.outer_migrate()
                self.synchro()
                self.dump()
                # 广播让其他进程也结束
                if self.isFinish:
                    break
            # 组内交换,组间迁移可能导致引入新的结点，此时master收集适应度，而新结点等待同步
            # 会卡住，所以必须分开
            elif self.size > 1:
                self.inner_migrate()
            # 实际上只有0号进程获取了，对于其他进程record是None，不能直接求和除Size

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


class Master:
    def __init__(self, T, migrate_rate, wait, conf_file, input_file, output_file, slaver_file):
        self.inputFile = input_file  # 数据源
        self.outputFile = output_file  # 数据源
        self.confFile = conf_file  # 配置文件
        self.slaverFile = slaver_file
        self.position = []  # 坐标
        self.cityNum = 0
        self.T = T  # 运行代数
        self.t = 0  # 当前代数
        self.bestDistance = np.inf  # 最佳长度
        self.bestPath = []  # 最佳路径
        self.record = []  # 记录适应度变化
        self.groupSize = 0  # 组数
        self.migrateRate = migrate_rate  # 每个migrateRate代组间迁移一次
        self.isFinish = False  # 该种群是否不在变化
        self.wait = wait
        self.w = wait
        self.ipTable = {}
        self.commTable = {}
        self.size = 0  # 进程数

    def read_conf(self):
        new_ip = []
        with open(self.confFile) as infile:
            for line in infile:
                split = line.strip().split()
                ip = split[0]
                n = int(split[1])
                if ip not in self.ipTable:
                    self.size += n
                    self.groupSize += 1
                    self.ipTable[ip] = n
                    new_ip.append(ip)
        return new_ip

    def spawn(self, ip):
        info = MPI.Info.Create()
        info.Set('host', ip)
        n = self.ipTable[ip]
        comm = MPI.COMM_SELF.Spawn(sys.executable, [self.slaverFile], maxprocs=n, info=info)
        return comm

    def update_comm(self, new_ip: list):
        isInit = len(self.commTable) == 0
        needBalance = len(new_ip) > 0 and not isInit
        self.w = self.wait if needBalance else self.w
        self.isFinish = self.w <= 0 or self.t >= self.T
        n = 0
        for ip in new_ip:
            n += self.ipTable[ip]
        population = []
        # 对于原来的进程
        for comm in self.commTable.values():
            comm.bcast((self.groupSize, self.isFinish), root=MPI.ROOT)
            comm.bcast((n, self.size), root=MPI.ROOT)
            if needBalance:
                population_list = comm.gather(None, root=MPI.ROOT)
                for p in population_list:
                    population.extend(p)
        if needBalance:
            n = len(population) // n

        # 对于新加入的进程
        off = 0
        for i, ip in enumerate(new_ip):
            comm = self.spawn(ip)
            # 发送rank和初始化
            comm.bcast((len(self.commTable), isInit), root=MPI.ROOT)
            self.commTable[ip] = comm
            if isInit:
                loadList = comm.gather(None, root=MPI.ROOT)
                for rank, load in enumerate(loadList):
                    if not load:
                        comm.send(self.position, dest=rank)
            else:
                for j in range(self.ipTable[ip]):
                    comm.send(self.position, dest=j)
                    comm.send(population[off * n:(off + 1) * n], dest=j)
                    off += 1
            comm.bcast((self.groupSize, self.isFinish, self.migrateRate), root=MPI.ROOT)

    # 读取文件
    def read_data(self):
        with open(self.inputFile) as infile:
            for line in infile:
                split = line.strip().split()
                self.position.append([float(split[1]), float(split[2])])
        self.cityNum = len(self.position)

    def outer_migrate(self):
        ipList = list(self.commTable.keys())
        print('开始组间迁移')
        for i in range(1, len(ipList)):
            print(ipList[i - 1] + "->" + ipList[i])
            src_comm = self.commTable[ipList[i - 1]]
            dest_comm = self.commTable[ipList[i]]
            population = src_comm.recv(source=0)
            dest_comm.send(population, dest=0)
        src_comm = self.commTable[ipList[-1]]
        dest_comm = self.commTable[ipList[0]]
        print(ipList[-1] + "->" + ipList[0])
        population = src_comm.recv(source=0)
        dest_comm.send(population, dest=0)

    def gather_fitness(self):
        records = []
        for comm in self.commTable.values():
            records.extend(comm.gather(None, root=MPI.ROOT))  # 实际上只有0号进程获取了，对于其他进程record是None，不能直接求和除Size
        self.record.append(np.sum(records) / self.size)

    def load_balance(self):
        pass

    def get_all_best(self):
        print("当前代数:" + str(self.t))
        best_list = []
        for comm in self.commTable.values():
            best_list.extend(comm.gather(None, root=MPI.ROOT))
        best_list.sort(key=lambda x: x[0])
        bestDistance = best_list[0][0]
        bestPath = best_list[0][1]
        if bestDistance < self.bestDistance:
            self.bestDistance = bestDistance
            self.bestPath = bestPath
            self.w = self.wait
        else:
            self.w -= 1
        print("最优距离:" + str(self.bestDistance))

    def synchro(self):
        self.get_all_best()
        new_ip = self.read_conf()
        self.update_comm(new_ip)

    def master_run(self):
        self.read_data()
        new_ip = self.read_conf()
        self.update_comm(new_ip)

        while True:
            self.t += 1
            self.gather_fitness()
            # 组间迁移后master收集最优结果
            if self.t != 0 and self.t % self.migrateRate == 0:
                if self.groupSize > 1:
                    self.outer_migrate()
                self.synchro()
                if self.isFinish:
                    break
        # 搜集其他种群最好的个体
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
