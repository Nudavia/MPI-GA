# -*-coding:utf-8-*-
from time import time

from ga import Master

if __name__ == '__main__':
    t1 = time()
    master = Master(T=1000,
                    migrate_rate=100,
                    wait=3,
                    conf_file="nodes.conf",
                    input_file="data/st70.txt",
                    output_file="result/st70.txt",
                    slaver_file='slaver.py')
    master.master_run()
    t2 = time()
    print("耗时:" + str(t2 - t1))
    master.show()
