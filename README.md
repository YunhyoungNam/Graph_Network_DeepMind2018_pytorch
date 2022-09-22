# Pyorch implementation of Graph Nets by Google DeepMind

[Graph Nets](https://github.com/deepmind/graph_nets) is DeepMind's library for
building graph networks in Tensorflow and Sonnet.

To learn more about graph networks, see our arXiv paper: [Relational inductive
biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261).

#### Results comparison

Hyper-parameters are used as-is in the original code, but the inference results are from other examples.

##### Sorest_path

Results in the paper
![Result in the paper](https://github.com/YunhyoungNam/Graph_Network_DeepMind2018_pytorch/tree/main/images/shortest-path.png)
Results from pytorch implementation
![Result pytorch code](https://github.com/YunhyoungNam/Graph_Network_DeepMind2018_pytorch/tree/main/images/shortest-path-pytorch.png)

##### Sort

Results in the paper
![Result in the paper](https://github.com/YunhyoungNam/Graph_Network_DeepMind2018_pytorch/tree/main/images/sort.png)
Results from pytorch implementation
![Result pytorch code](https://github.com/YunhyoungNam/Graph_Network_DeepMind2018_pytorch/tree/main/images/sort-pytorch.png)

##### Physics

Results in the paper
![Result in the paper](https://github.com/YunhyoungNam/Graph_Network_DeepMind2018_pytorch/tree/main/images/physics.png)
Results from pytorch implementation
![Result pytorch code](https://github.com/YunhyoungNam/Graph_Network_DeepMind2018_pytorch/tree/main/images/physics-pytorch.png)

## Usage example

In order to define graph network, initial graph tuple has to be created.
Because each block have the initial graph as the argment together with network information (e.g num_layers, dimension, activation ...)

Different from the sonet of tensorflow implementation, network information of each block is defined using dictionary like belows.
Thus, It can be set seperately for each block of every network.

```python
#network information
net_info = {'dim' : 16,
            'n_blk' : 2,
            'norm' : 'ln',
            'activ' : 'relu',
            'in_dim' : 'none',
            'out_dim' : 16,
            'norm_final' : 'ln',
            'activ_final' : 'relu',
            'init_weight' : 'normal'}
```

## Demo codes

To reproduce, run the following demo_shortestpaht.py , \_sort.py, & \_physics.py.
