

# please download https://github.com/QwenLM/Qwen2

```
cd examples/sft/finetune.sh


# the --data option is customised by us, you can change the data path as you like 

LOGLEVEL=INFO NNODES=2 NODE_RANK=0 MASTER_ADDR=172.31.255.252 bash finetune.sh  --data  sameprogram23_1v0_2w_tosem


LOGLEVEL=INFO NNODES=2 NODE_RANK=1 MASTER_ADDR=172.31.255.252 MASTER_PORT=6006 bash finetune.sh  --data  sameprogram23_1v0_2w_tosem


```

