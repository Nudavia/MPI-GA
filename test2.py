# -*-coding:utf-8-*-
# spawn_slave.py

"""
Demonstrates the usage of Get_parent, Scatter, Gather, Disconnect.
"""

import socket

from mpi4py import MPI

Comm = MPI.COMM_WORLD
if __name__ == '__main__':
    # get the parent intercommunicator
    # comm = MPI.Comm.Get_parent()
    comm = MPI.COMM_WORLD
    # print('slave: rank %d of %d' % (comm.rank, comm.size))
    # print('slave: rank %d of %s' % (comm.rank, socket.gethostname()))

    # receive data from master process
    # recv_buf = np.array(0, dtype='i')
    # recv_buf = comm.scatter(None, root=0)
    # print(MPI.ROOT)
    # recv_buf = comm.recv(source=0)
    # print('slave: rank %d receives %d' % (comm.rank, recv_buf))

    # increment the received data
    # recv_buf += 1

    # send the incremented data to master
    # comm.send(recv_buf, dest=0)
    comm.send(comm.rank, dest=(comm.rank + 1) % comm.size)
    print(comm.recv(source=(comm.rank - 1) % comm.size))
    print(socket.gethostname() + '--' + str(comm.rank))
    # while True:
    # print(socket.gethostname())
    # dest = (comm.rank + 1) % 2
    # src = (comm.rank - 1 + 2) % 2
    # Comm.send(recv_buf, dest=dest)
    # s = Comm.recv(source=src)

    # print('slave: rank %d sends %d' % (comm.rank, recv_buf))
    # print('slave: rank %d sends %d' % (comm.rank, s))
    # disconnect and free comm
    # comm.Disconnect()
