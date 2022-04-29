# -*-coding:utf-8-*-
from mpi4py import MPI
import sys
import socket

if __name__ == '__main__':
    # print(cpu_percent(2))
    # print(cpu_count())
    # print(socket.gethostname())
    # print(virtual_memory().percent)
    # print(virtual_memory().free)
    """
    Demonstrates the usage of Spawn, Scatter, Gather, Disconnect.

    Run this with 1 process like:
    $ mpiexec -n 1 python spawn_master.py
    # or
    $ python spawn_master.py
    """
    info = MPI.Info.Create()
    info.Set('host', 'domain04')
    # info.Set('localonly', 'true')
    # create two new processes to execute spawn_slave.py
    comm = MPI.COMM_WORLD.Spawn(sys.executable, ['test2.py'], maxprocs=2, info=info)
    # info = MPI.Info.Create()
    # i = 0
    # while 10000000000:
    #     i += 1
    print(socket.gethostname())
    # print(info.get('add-host'))
    # print('master: rank %d of %d' % (comm.rank, comm.size))

    # scatter [1, 2] to the two new processes
    # send_buf = [1, 2]
    # comm.scatter(send_buf, root=MPI.ROOT)
    # comm.send(send_buf[0], dest=0)
    # comm.send(send_buf[1], dest=1)
    # print('master: rank %d sends %s' % (comm.rank, send_buf))

    # gather data from the two new processes
    # recv_buf = np.array([0, 0], dtype='i')
    # recv_buf = comm.gather(None, root=MPI.ROOT)
    # recv_buf = []
    # recv_buf.append(comm.recv(source=0))
    # recv_buf.append(comm.recv(source=1))
    # print(comm.gather(None, root=MPI.ROOT))
    # print('master: rank %d receives %s' % (comm.rank, recv_buf))
    # disconnect and free comm
    # comm.Disconnect()

"""
This is the interface that allows for creating nested lists.
You should not implement it, or speculate about its implementation
"""
