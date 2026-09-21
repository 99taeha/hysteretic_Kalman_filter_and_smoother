
# hysteretic_Kalman_filter_and_smoother

## Hardware

Computations in this repository were run on the following machine.

| Component   | Specification                                                          |
| ----------- | ---------------------------------------------------------------------- |
| CPU         | AMD Ryzen 7 9700X (8 cores / 16 threads, L2 8 MiB, L3 32 MiB, AVX-512) |
| Memory      | 32 GB DDR5-5600 (2 × 16 GB, SK Hynix HMCG78AGBUA081N)                 |
| GPU         | NVIDIA GeForce RTX 4080 SUPER (16 GB VRAM)                             |
| Motherboard | MSI MAG B650M MORTAR WIFI (MS-7D76)                                    |
| Storage     | 1 TB virtual disk (WSL2)                                               |

## Software environment

| Item           | Version                                                     |
| -------------- | ----------------------------------------------------------- |
| Host OS        | Windows 11 Pro (build 10.0.26200)                           |
| Linux (WSL2)   | Ubuntu 24.04.2 LTS, kernel 6.6.87.2-microsoft-standard-WSL2 |
| WSL2 resources | 16 logical CPUs, ~15 GiB RAM, 4 GiB swap                    |
| NVIDIA driver  | 591.86 (CUDA 13.1)                                          |
| Python         | 3.10.9                                                      |
| NumPy          | 1.24.2                                                      |
| SciPy          | 1.10.1                                                      |
| Matplotlib     | 3.8.4                                                       |
