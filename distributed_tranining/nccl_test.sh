# please follow https://github.com/NVIDIA/nccl-tests/issues/27

```
make MPI=1 MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl

mpirun -H host1,host2 -np 2 ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 4

```
