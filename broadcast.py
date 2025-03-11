import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank_id, size):
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank_id
    print('before broadcast',' Rank ', rank_id, ' has data ', tensor)
    dist.broadcast(tensor, src = 0)
    print('after broadcast',' Rank ', rank_id, ' has data ', tensor)


def init_process(rank_id, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank_id, world_size=size)
    fn(rank_id, size)


if __name__ == "__main__":
    size = 4  # Number of processes
    mp.spawn(init_process, args=(size, run), nprocs=size, join=True)