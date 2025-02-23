import pathlib
import time

import torch

from dinora.models import model_selector
from dinora.models.alphanet import AlphaNet


def benchmark_throughput(
    model: AlphaNet, batch_size: int, warmup_iters: int, test_iters: int
):
    input_tensor = torch.randn(batch_size, 18, 8, 8, device=model.device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_tensor)

    torch.cuda.synchronize() if model.device.type == "cuda" else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(test_iters):
            _ = model(input_tensor)

    torch.cuda.synchronize() if model.device.type == "cuda" else None
    elapsed_time = time.time() - start_time

    throughput = (batch_size * test_iters) / elapsed_time
    return throughput


def bench_torch(
    model: AlphaNet,
    warmup_iters: int,
    test_iters: int,
    batch_sizes: list[int],
):
    # This tunes conv algorithm, increases throughput, but makes latency worse
    # torch.backends.cudnn.benchmark = True

    model.eval()

    for i, batch_size in enumerate(batch_sizes):
        throughput = benchmark_throughput(model, batch_size, warmup_iters, test_iters)
        print(f"{i}: Batch Size: {batch_size}, Throughput: {throughput:.2f} images/sec")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("weights_path", type=pathlib.Path)
    parser.add_argument("--warmup_iters", default=10, type=int)
    parser.add_argument("--test_iters", default=30, type=int)
    parser.add_argument("--batch_power", default=10, type=int)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_selector("alphanet", args.weights_path, device.type)
    assert isinstance(model, AlphaNet)

    bench_torch(
        model=model,
        warmup_iters=args.warmup_iters,
        test_iters=args.test_iters,
        batch_sizes=[2**n for n in range(args.batch_power)],
    )
