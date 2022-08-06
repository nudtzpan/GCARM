This is the code for our paper in TOIS: [Graph Co-Attentive Session-based Recommendation](https://dl.acm.org/doi/abs/10.1145/3486711).

## Requirements

- Python 3.8
- PyTorch 1.8.0


## Quick Start
**Firstly**, generate the global graph and local graph:

### diginetica dataset
```bash
python global_graph.py --dataset diginetica --n_num 2
python local_graph.py --dataset diginetica
```

### gowalla dataset
```bash
python global_graph.py --dataset gowalla --n_num 2
python local_graph.py --dataset gowalla
```

**Secondly**, train the model and predict:

### diginetica dataset
```bash
python main.py --dataset diginetica --batchSize 512 --hiddenSize 128 --l2 0 --drop_prob 0.25 --n_num 2 --hop 2 --neg_frac 128
```

### gowalla dataset
```bash
python main.py --dataset gowalla --batchSize 512 --hiddenSize 128 --l2 0 --drop_prob 0.25 --n_num 2 --hop 2 --neg_frac 128
```

## Citation

Please cite our papers if you use the code:

```
@article{DBLP:journals/tois/PanCCC22,
  author    = {Zhiqiang Pan and Fei Cai and Wanyu Chen and Honghui Chen},
  title     = {Graph Co-Attentive Session-based Recommendation},
  journal   = {{ACM} Trans. Inf. Syst.},
  volume    = {40},
  number    = {4},
  pages     = {67:1--67:31},
  year      = {2022}
}
```
