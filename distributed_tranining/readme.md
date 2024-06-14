# This is distributed traning test script and using nccl-test to measusutre bottleneck in Multi Node Distributed Training

## Distributed Traning

please refer to the script `bash distribut_run.sh`, we will update it later. 

## Measure the high bandwidth and low latency

please refer to [nccl-tests](https://github.com/NVIDIA/nccl-tests) 

This script is `bash nccl_test.sh` is about how we test bandwith in bitdeer.ai platform, you can change the host . 

```
# cat host 
csl-dgx1v-1 slots=12
csl-server3 slots=12
```

```
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS

export OMPI_MCA_btl_base_verbose=100
export OMPI_MCA_btl_tcp_verbose=100

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xxxx/openmpi/lib

#mpirun -np 2 --hostfile hosts -x NCCL_DEBUG=INFO  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 2 


#all_reduce_perf
# Node1 configuration
#if [[ $(hostname) == "csl-dgx1v-1" ]]; then
#         export OMPI_MCA_btl_tcp_if_include=enp1s0f0
#fi

# Node2 configuration
#if [[ $(hostname) == "csl-server3" ]]; then
#         export OMPI_MCA_btl_tcp_if_include=enp96s0f1
#fi




#/home/csladmin/openmpi/bin/mpirun  --allow-run-as-root --mca btl_tcp_if_exclude docker0,enp132s0np0,veth2867,veth03d5,veth05cf,veth05dc,veth07ac,veth1241,veth14b4,veth1abb,veth2d46,veth2de9,veth3210,veth340f,veth5122,veth56b7,veth59e2,veth5d7e,veth720b,veth80f4,veth89cd,veth8ee0,vetha132,vethab9c,vethb091,vethc5f8,vethd55b,vethdbea,vethdff4,vethe481,vethf3db,lo  --hostfile hosts  -np 16  -x NCCL_DEBUG=INFO  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

#/home/csladmin/openmpi/bin/mpirun  --allow-run-as-root  -np 12  --mca plm_rsh_args "-p 5000" -mca btl_base_verbose 100 -bind-to none -map-by slot -x NCCL_DEBUG=INFO  --hostfile hosts  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1 

#/home/csladmin/openmpi/bin/mpirun  --allow-run-as-root  -np 12 --mca btl_tcp_if_include enp64s0f0,enp132s0np0  -mca btl_base_verbose 100 -bind-to none -map-by slot -x NCCL_DEBUG=INFO  --hostfile hosts  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1 


#/home/csladmin/openmpi/bin/mpirun  --allow-run-as-root  --hostfile hosts -x MPI=1 -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH -x NCCL_IB_HCA=mlx5_0 -x NCCL_NET_GDR_LEVEL=5 -x NCCL_SHM_DISABLE=1 -x NCCL_IB_MERGE_VFS=0 -x NCCL_IGNORE_DISABLED_P2P=1 -x NCCL_IB_DISABLE=0 -np 12   -x NCCL_DEBUG=INFO  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1 


#/home/csladmin/openmpi/bin/mpirun  --allow-run-as-root  --mca btl_tcp_if_exclude  192.168.128.100/120  --hostfile hosts  -np 16  -x NCCL_DEBUG=INFO  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

/home/csladmin/openmpi/bin/mpirun  --allow-run-as-root  --mca btl_tcp_if_include enp64s0f0 --  --hostfile hosts  -np 16  -x NCCL_DEBUG=INFO  ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```
