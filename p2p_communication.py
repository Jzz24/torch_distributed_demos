import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run1(rank_id, size):
    tensor = torch.zeros(1)
    if rank_id == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
        print('after send, Rank ', rank_id, ' has data ', tensor[0])
        dist.recv(tensor=tensor, src=1)
        print('after recv, Rank ', rank_id, ' has data ', tensor[0])
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
        print('after recv, Rank ', rank_id, ' has data ', tensor[0])
        tensor += 1
        dist.send(tensor=tensor, dst=0)
        print('after send, Rank ', rank_id, ' has data ', tensor[0])


def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])


def init_process(rank_id, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank_id, world_size=size)
    fn(rank_id, size)


if __name__ == "__main__":
    size = 2  # Number of processes
    mp.spawn(init_process, args=(size, run), nprocs=size, join=True)