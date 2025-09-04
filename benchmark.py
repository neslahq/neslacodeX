import os
import hydra
import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from src.codex.model import Codex, DataloaderLite, get_lr


def train_step(step, device, ddp, optimizer, gradient_accumulation_steps, dataloader, model, config):
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(gradient_accumulation_steps):
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        # Autocast increases computation on the device because it adds dtype conversion operations that wouldn't exist otherwise. But the is little compared to the increase in flops and speed of loading data it provides. 
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            torch.cuda.synchronize()
            t_hat = time.perf_counter()
            loss = loss / gradient_accumulation_steps
            loss_accum += loss.detach()
        # we don't want ddp to sync gradients every micro_step
        #TODO: check if neccesary to use gradient scaling here
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        loss.backward()
    if ddp:
        # get loss_accum from all processes and average them, so we print the average loss for all processes and not just for rank 0
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    lr = get_lr(step, config)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    if device == "mps":
        torch.mps.synchronize()
    
    return loss_accum, norm, lr, t_hat
    



@hydra.main(config_name="benchmark_config.yaml")
def benchmark(config):
    ddp = config.train.ddp

    assert torch.cuda.is_available(), "For ddp training, cuda is required"
    dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK"))
    local_rank = int(os.environ.get("LOCAL_RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    master_process = rank == 0

    torch.set_float32_matmul_precision(config.model.matmul_precision)


    dataloader = DataloaderLite(8, 1024, config.data, rank, world_size)
    B, T = dataloader.B, dataloader.T
    total_batch_size = config.train.total_batch_size

    assert (
        total_batch_size % (B * T * world_size) == 0
    ), "total_batch_size must be divisible by the batch size x sequence length"

    if config.train.gradient_accumulation:
        gradient_accumulation_steps = total_batch_size // (B * T * world_size)
    else:
        gradient_accumulation_steps = 1

    model = Codex(config.model)
    model.to(device)

    
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizer(device)

    # for epoch in range(config.train.epochs):
    for step in range(config.train.num_warmups):
        t0 = time.perf_counter()
        loss_accum, norm, lr , t_hat = train_step(step, device, ddp, optimizer, gradient_accumulation_steps, dataloader, model, config)
        t1 = time.perf_counter()
        fdt = (t_hat - t0) * 1000
        dt = (t1 - t0) * 1000
        toks_sec = (B * T * gradient_accumulation_steps * world_size) / (t1 - t0)
        if rank == 0:
            print(
                f"Step: {step}, lr: {lr}, loss: {loss_accum.item()}, fwd_time: {fdt}ms, time: {dt}ms, toks/sec: {toks_sec}, norm: {norm}"
            )

    fdt_m = []
    dt_m = []
    for step in range(config.train.max_steps):
        t0 = time.perf_counter()
        loss_accum, norm, lr, t_hat = train_step(step, device, ddp, optimizer, gradient_accumulation_steps, dataloader, model, config)
        t1 = time.perf_counter()
        fdt = (t_hat - t0) * 1000
        dt = (t1 - t0) * 1000
        toks_sec = (B * T * gradient_accumulation_steps * world_size) / (t1 - t0)
        if rank == 0:
            fdt_m.append(fdt)
            dt_m.append(dt)
            print(
                f"Step: {step}, lr: {lr}, loss: {loss_accum.item()}, fwd_time: {fdt}ms, time: {dt}ms, toks/sec: {toks_sec}, norm: {norm}"
            )
    if rank == 0:
        fwd_tm = torch.tensor(fdt_m).std()
        total_time = torch.tensor(dt_m).std()
        print(f"fwd_std: {fwd_tm.item()}, bwd_std: {total_time.item()}")
        
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    benchmark()