# MPI-GA
分布式MPI解决TSP问题，可动态添加结点以及容错
该项目可以单节点运行也可以在多结点上运行
通过master.py 启动MPI程序，然后会根据nodes.conf文件的配置在对应结点上运行MPI并行计算
计算过程中可以往文件里添加结点，实现动态添加并自动分配数据
计算过程中会保存每一轮的数据，如果出错中断了，再次激动会继续计算，从而提供了容错
记得每次启动前清空checkpoint目录下的文件