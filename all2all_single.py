# import torch
# import torch.distributed as dist
# import os
# import torch.multiprocessing as mp

# def run(rank, size):
#     """分布式函数，执行 all_to_all_single 操作"""
#     # 初始化分布式进程组
#     if not dist.is_initialized():
#         dist.init_process_group(backend='nccl', init_method='env://', world_size=size, rank=rank)
    
#     # 设置当前进程的 GPU 设备
#     torch.cuda.set_device(rank)
    
#     # 定义每个 rank 的输入张量
#     input_tensors = {
#         0: torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64, device='cuda'),
#         1: torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.int64, device='cuda'),
#         2: torch.tensor([20, 21, 22, 23, 24], dtype=torch.int64, device='cuda'),
#         3: torch.tensor([30, 31, 32, 33, 34, 35, 36], dtype=torch.int64, device='cuda')
#     }
    
#     # 定义每个 rank 的输入分割配置
#     input_splits_list = {
#         0: [2, 2, 1, 1],
#         1: [3, 2, 2, 2],
#         2: [2, 1, 1, 1],
#         3: [2, 2, 2, 1]
#     }
    
#     # 定义每个 rank 的输出分割配置
#     output_splits_list = {
#         0: [2, 3, 2, 2],
#         1: [2, 2, 1, 2],
#         2: [1, 2, 1, 1],
#         3: [1, 2, 1, 1]
#     }
    
#     # 获取当前 rank 的输入张量和分割配置
#     input_tensor = input_tensors[rank]
#     input_splits = input_splits_list[rank]
#     output_splits = output_splits_list[rank]
    
#     # 计算输出张量的大小并创建空张量
#     output_size = sum(output_splits)
#     output_tensor = torch.empty(output_size, dtype=torch.int64, device='cuda')
    
#     # 执行 all_to_all_single 操作
#     dist.all_to_all_single(
#         output_tensor,              # 输出张量
#         input_tensor,               # 输入张量
#         output_split_sizes=output_splits,  # 输出分割配置
#         input_split_sizes=input_splits     # 输入分割配置
#     )
    
#     # 打印当前 rank 的输出结果
#     print(f"Rank {rank} output: {output_tensor.tolist()}")

# def init_process(rank, size, fn, backend='nccl'):
#     """ Initialize the distributed environment. """
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '29500'
#     if not dist.is_initialized():
#         dist.init_process_group(backend, rank=rank, world_size=size)
#     fn(rank, size)
#     if dist.is_initialized():
#         dist.destroy_process_group()

# if __name__ == "__main__":
#     size = 4  # Number of processes
#     mp.set_start_method("spawn")
#     mp.spawn(init_process, args=(size, run), nprocs=size, join=True)


import torch
import torch.distributed as dist
import os
import torch.multiprocessing as mp

def run(rank, size):
    """分布式函数，执行 all_to_all_single 操作"""
    # 初始化分布式进程组
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://', world_size=size, rank=rank)
    
    # 设置当前进程的 GPU 设备
    torch.cuda.set_device(rank)
    
    # 定义每个 rank 的输入张量
    input_tensors = {
        0: torch.tensor([2, 3, 1, 2], dtype=torch.int64, device='cuda'),
        1: torch.tensor([4, 1, 2, 3], dtype=torch.int64, device='cuda')
    }
    
    # 获取当前 rank 的输入张量
    input_tensor = input_tensors[rank]
    
    # 创建一个新的张量用于存储每个进程的 token 数量
    tokens_per_expert_group = input_tensor.new_empty(input_tensor.shape[0])
    
    # 使用 all_to_all_single 进行通信
    dist.all_to_all_single(tokens_per_expert_group, input_tensor)
    
    # 打印当前 rank 的输出结果
    print(f"Rank {rank} tokens_per_expert_group: {tokens_per_expert_group.tolist()}")

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    if not dist.is_initialized():
        dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    size = 2  # Number of processes
    mp.set_start_method("spawn")
    mp.spawn(init_process, args=(size, run), nprocs=size, join=True)