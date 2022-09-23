# Pyorch implementation of Graph Nets by Google DeepMind

[Graph Nets](https://github.com/deepmind/graph_nets) is DeepMind's library for
building graph networks in Tensorflow and Sonnet.

To learn more about graph networks, see our arXiv paper: [Relational inductive
biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261).

#### Results comparison

Hyper-parameters are used as-is in the original code, but the inference results are from other examples.

##### Shortest_path

Results in the paper
![shortest-path](https://user-images.githubusercontent.com/66901621/191691901-886fb80e-42b3-47a8-81bd-ec6ad413049a.png)

Results from pytorch implementation
![shortest-path-pytorch](https://user-images.githubusercontent.com/66901621/191692240-3af9e356-0ec9-4c95-ac82-a9d441c57661.png)

##### Sort

Results in the paper
![sort](https://user-images.githubusercontent.com/66901621/191694069-fef0f701-b949-4758-92b6-c8b019bf39d8.png)

Results from pytorch implementation
![sort-pytorch-](https://user-images.githubusercontent.com/66901621/191695070-0f23255c-c9d1-483d-8a9b-a3bda04e4000.png)

##### Physics

Results in the paper
![physics](https://user-images.githubusercontent.com/66901621/191692555-f576fdc9-b603-4aba-8a52-85ffe6194b81.png)

Results from pytorch implementation
![physics-pytorch](https://user-images.githubusercontent.com/66901621/191692658-c1e992bf-a0b8-4e44-a94a-8eff850e43ce.png)

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
