# -*-coding:utf-8-*-
from time import time

from ga import Slaver

if __name__ == '__main__':
    slaver = Slaver(M=200,
                    p_cross=0.7,
                    p_mutate=0.05,
                    max_checkpoint=2)
    slaver.slaver_run()
