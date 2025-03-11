# # import os
# # import torch
# # import torch.distributed as dist
# # import torch.multiprocessing as mp

# # def run(rank, size):
# #     # 设置每个进程使用的 GPU 设备
# #     torch.cuda.set_device(rank)
    
# #     # 每个进程初始化自己的输入张量
# #     input_tensor = torch.arange(size * 2, dtype=torch.int64, device='cuda') + rank * size * 2
# #     output_tensor = torch.zeros(size * 2, dtype=torch.int64, device='cuda')
    
# #     # 定义每个进程发送和接收的元素数量
# #     send_counts = [2] * size
# #     recv_counts = [2] * size
    
# #     print(f'Rank {rank} before all_to_all:\nInput tensor:\n{input_tensor}\nOutput tensor:\n{output_tensor}\n')
    
# #     # 执行 all_to_all 操作
# #     dist.all_to_all(
# #         list(output_tensor.split(recv_counts)),
# #         list(input_tensor.split(send_counts))
# #     )
    
# #     print(f'Rank {rank} after all_to_all:\nInput tensor:\n{input_tensor}\nOutput tensor:\n{output_tensor}\n')

# # def init_process(rank, size, fn, backend='nccl'):
# #     """ Initialize the distributed environment. """
# #     os.environ['MASTER_ADDR'] = '127.0.0.1'
# #     os.environ['MASTER_PORT'] = '29500'
# #     dist.init_process_group(backend, rank=rank, world_size=size)
# #     fn(rank, size)
# #     dist.destroy_process_group()

# # if __name__ == "__main__":
# #     size = 3  # Number of processes
# #     mp.set_start_method("spawn")
# #     mp.spawn(init_process, args=(size, run), nprocs=size, join=True)


# import torch
# import torch.distributed as dist
# import os

# def split_tensor(tensor, splits):
#     """根据 splits 将 tensor 分割成多个小张量"""
#     return torch.split(tensor, splits)

# def run(rank, size):
#     """分布式函数，执行 all_to_all 操作"""
#     # 初始化分布式进程组
#     dist.init_process_group(backend='nccl', init_method='env://', world_size=size, rank=rank)
    
#     # 设置当前进程的 GPU 设备
#     torch.cuda.set_device(rank)
    
#     # 定义每个 rank 的输入张量（在 GPU 上）
#     input_tensors = {
#         0: torch.tensor([0, 1, 2, 3, 4, 5], device='cuda'),
#         1: torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18], device='cuda'),
#         2: torch.tensor([20, 21, 22, 23, 24], device='cuda'),
#         3: torch.tensor([30, 31, 32, 33, 34, 35, 36], device='cuda')
#     }
    
#     # 定义每个 rank 的 input_splits
#     input_splits_list = {
#         0: [2, 2, 1, 1],
#         1: [3, 2, 2, 2],
#         2: [2, 1, 1, 1],
#         3: [2, 2, 2, 1]
#     }
    
#     # 定义每个 rank 的 output_splits
#     output_splits_list = {
#         0: [2, 3, 2, 2],
#         1: [2, 2, 1, 2],
#         2: [1, 2, 1, 1],
#         3: [1, 2, 1, 2]
#     }
    
#     # 获取当前 rank 的输入张量和 splits
#     input_tensor = input_tensors[rank]
#     input_splits = input_splits_list[rank]
#     output_splits = output_splits_list[rank]
    
#     # 根据 input_splits 分割输入张量
#     input_tensor_list = list(split_tensor(input_tensor, input_splits))
    
#     # 创建输出张量列表，根据 output_splits
#     output_tensor_list = [torch.empty(split, dtype=torch.int64, device='cuda') for split in output_splits]
    
#     # 执行 all_to_all 操作
#     dist.all_to_all(output_tensor_list, input_tensor_list)
    
#     # 打印当前 rank 的输出结果
#     print(f"Rank {rank} output: {[t.tolist() for t in output_tensor_list]}")
    
#     # 清理分布式进程组
#     dist.destroy_process_group()

# if __name__ == "__main__":
#     # 设置进程总数（世界大小）
#     size = 4
#     # 获取当前进程的 rank
#     rank = int(os.environ['RANK'])
#     # 运行分布式函数
#     run(rank, size)


import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def split_tensor(tensor, splits):
    """根据 splits 将 tensor 分割成多个小张量"""
    return torch.split(tensor, splits)

def run(rank, size):
    """分布式函数，执行 all_to_all 操作"""
    # 初始化分布式进程组
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://', world_size=size, rank=rank)
    
    # 设置当前进程的 GPU 设备
    torch.cuda.set_device(rank)
    
    # 定义每个 rank 的输入张量（在 GPU 上）
    input_tensors = {
        0: torch.tensor([0, 1, 2, 3, 4, 5], device='cuda'),
        1: torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18], device='cuda'),
        2: torch.tensor([20, 21, 22, 23, 24], device='cuda'),
        3: torch.tensor([30, 31, 32, 33, 34, 35, 36], device='cuda')
    }
    
    # 定义每个 rank 的 input_splits
    input_splits_list = {
        0: [2, 2, 1, 1],
        1: [3, 2, 2, 2],
        2: [2, 1, 1, 1],
        3: [2, 2, 2, 1]
    }
    
    # 定义每个 rank 的 output_splits
    output_splits_list = {
        0: [2, 3, 2, 2],
        1: [2, 2, 1, 2],
        2: [1, 2, 1, 1],
        3: [1, 2, 1, 2]
    }
    
    # 获取当前 rank 的输入张量和 splits
    input_tensor = input_tensors[rank]
    input_splits = input_splits_list[rank]
    output_splits = output_splits_list[rank]
    
    # 根据 input_splits 分割输入张量
    input_tensor_list = list(split_tensor(input_tensor, input_splits))
    
    # 创建输出张量列表，根据 output_splits
    output_tensor_list = [torch.empty(split, dtype=torch.int64, device='cuda') for split in output_splits]
    
    # 执行 all_to_all 操作
    dist.all_to_all(output_tensor_list, input_tensor_list)
    
    # 打印当前 rank 的输出结果
    print(f"Rank {rank} output: {[t.tolist() for t in output_tensor_list]}")

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    if not dist.is_initialized():
        dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    size = 4  # Number of processes
    mp.set_start_method("spawn")
    mp.spawn(init_process, args=(size, run), nprocs=size, join=True)