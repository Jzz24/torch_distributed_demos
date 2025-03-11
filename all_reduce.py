# import os
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp

# def run(rank_id, size):
#     tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank_id
#     print('before reudce',' Rank ', rank_id, ' has data ', tensor)
#     dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
#     print('after reudce',' Rank ', rank_id, ' has data ', tensor)


# def init_process(rank_id, size, fn, backend='gloo'):
#     """ Initialize the distributed environment. """
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '29500'
#     dist.init_process_group(backend, rank=rank_id, world_size=size)
#     fn(rank_id, size)


# if __name__ == "__main__":
#     size = 4
#     processes = []
#     mp.set_start_method("spawn")
#     for rank in range(size):
#         p = mp.Process(target=init_process, args=(rank, size, run))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank_id, size):
    # 设置当前进程使用的 GPU 设备
    torch.cuda.set_device(rank_id)
    
    tensor = torch.arange(2, dtype=torch.int64).cuda() + 1 + 2 * rank_id
    print(f'Rank {rank_id} before all_reduce: {tensor}')
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f'Rank {rank_id} after all_reduce: {tensor}')

    # 销毁进程组
    dist.destroy_process_group()

def init_process(rank_id, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank_id, world_size=size)
    fn(rank_id, size)

if __name__ == "__main__":
    size = 4
    mp.set_start_method("spawn")
    mp.spawn(init_process, args=(size, run), nprocs=size, join=True)