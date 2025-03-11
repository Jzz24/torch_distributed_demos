import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank_id, size):
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank_id
    print('before gather',' Rank ', rank_id, ' has data ', tensor)
    if rank_id == 0:
        gather_list = [torch.zeros(2, dtype=torch.int64) for _ in range(4)]
        dist.gather(tensor, dst = 0, gather_list=gather_list)
        print('after gather',' Rank ', rank_id, ' has data ', tensor)
        print('gather_list:', gather_list)
    else:
        dist.gather(tensor, dst = 0)
        print('after gather',' Rank ', rank_id, ' has data ', tensor)

def init_process(rank_id, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank_id, world_size=size)
    fn(rank_id, size)


if __name__ == "__main__":
    size = 4
    mp.spawn(init_process, args=(size, run), nprocs=size, join=True)