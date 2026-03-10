import pathlib
import time

import numpy as np
import onnxruntime as ort


def benchmark_throughput(
    session: ort.InferenceSession, batch_size: int, warmup_iters: int, test_iters: int
) -> float:
    input_data = np.random.randn(batch_size, 18, 8, 8).astype(np.float32)

    # Warm-up
    for _ in range(warmup_iters):
        _ = session.run(None, {"input": input_data})

    start_time = time.time()

    for _ in range(test_iters):
        _ = session.run(None, {"input": input_data})

    elapsed_time = time.time() - start_time

    throughput = (batch_size * test_iters) / elapsed_time
    return throughput


def bench_onnx(
    session: ort.InferenceSession,
    warmup_iters: int,
    test_iters: int,
    batch_sizes: list[int],
) -> None:
    for i, batch_size in enumerate(batch_sizes):
        throughput = benchmark_throughput(session, batch_size, warmup_iters, test_iters)
        print(f"{i}: Batch Size: {batch_size}, Throughput: {throughput:.2f} boards/sec")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("weights_path", type=pathlib.Path)
    parser.add_argument("--warmup_iters", default=10, type=int)
    parser.add_argument("--test_iters", default=30, type=int)
    parser.add_argument("--batch_power", default=10, type=int)

    args = parser.parse_args()

    providers = ["CUDAExecutionProvider"]
    session = ort.InferenceSession(args.weights_path, providers=providers)
    batch_sizes = [2**n for n in range(args.batch_power)]

    bench_onnx(
        session=session,
        warmup_iters=args.warmup_iters,
        test_iters=args.test_iters,
        batch_sizes=batch_sizes,
    )
