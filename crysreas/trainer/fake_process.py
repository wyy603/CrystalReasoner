import torch
import time
import os
import argparse

def burn_gpu(device_id, num=30, matrix_size=4000, delay=0.01):
    """
    通过矩阵乘法模拟负载。
    target_utilization: 目标利用率 (0.7-0.8)
    memory_gb: 占用多少显存，避免主程序因 OOM 崩溃
    """
    print(f"Starting GPU load on device: {device_id}")
    device = torch.device(f"cuda:{device_id}")

    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)

    try:
        while True:
            for _ in range(int(num)):
                torch.matmul(a, b)

            torch.cuda.synchronize()
            
            time.sleep(delay)
    except KeyboardInterrupt:
        print("Stopped by user")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="GPU ID (0 or 1)")
    parser.add_argument("--num", type=int, default=5, help="num")
    parser.add_argument("--matrix_size", type=int, default=4000, help="matrix_size")
    parser.add_argument("--delay", type=float, default=0.01, help="delay")
    args = parser.parse_args()

    # 设置进程优先级为最低 (Linux/Unix)
    try:
        os.nice(19) 
    except:
        pass

    burn_gpu(args.device, args.num, args.matrix_size, args.delay)