# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 18:53:26 2022

@author: darkstar0983
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import time

from src import utils_np
from src import utils_pt

import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np

def create_graph_dicts_pt(num_examples, num_elements_min_max):
  """Generate graphs for training.

  Args:
    num_examples: total number of graphs to generate
    num_elements_min_max: a 2-tuple with the minimum and maximum number of
      values allowable in a graph. The number of values for a graph is
      uniformly sampled withing this range. The upper bound is exclusive, and
      should be at least 2 more than the lower bound.

  Returns:
    inputs: contains the generated random numbers as node values.
    sort_indices: contains the sorting indices as nodes. Concretely
      inputs.nodes[sort_indices.nodes] will be a sorted array.
    ranks: the rank of each value in inputs normalized to the range [0, 1].
  """

  num_elements = torch.from_numpy(np.random.uniform(low=num_elements_min_max[0],
                                  high=num_elements_min_max[1],
                                  size=[num_examples])).type(torch.int32)
  
  inputs_graphs = []
  sort_indices_graphs = []
  ranks_graphs = []
  for i in range(num_examples):
    #values = tf.random_uniform(shape=[num_elements[i]])
    values = torch.from_numpy(np.random.uniform(size=[num_elements[i]])).float()
    #sort_indices = tf.cast(tf.argsort(values, axis=-1), tf.float32)
    sort_indices = torch.argsort(values, dim=-1).type(torch.float32)
    ranks = torch.argsort(sort_indices, dim=-1).type(torch.float32) / (
        num_elements[i].type(torch.float32) - 1.0)
    
    #ranks = tf.cast(tf.argsort(sort_indices, axis=-1), tf.float32) / (
    #        tf.cast(num_elements[i], tf.float32) - 1.0)
    inputs_graphs.append({"nodes": values[:, None]})
    sort_indices_graphs.append({"nodes": sort_indices[:, None]})
    ranks_graphs.append({"nodes": ranks[:, None]})
  return inputs_graphs, sort_indices_graphs, ranks_graphs


def create_data_ops(batch_size, num_elements_min_max):
  """Returns graphs containing the inputs and targets for classification.

  Refer to create_data_dicts_tf and create_linked_list_target for more details.

  Args:
    batch_size: batch size for the `input_graphs`.
    num_elements_min_max: a 2-`tuple` of `int`s which define the [lower, upper)
      range of the number of elements per list.

  Returns:
    inputs_op: a `graphs.GraphsTuple` which contains the input list as a graph.
    targets_op: a `graphs.GraphsTuple` which contains the target as a graph.
    sort_indices_op: a `graphs.GraphsTuple` which contains the sort indices of
      the list elements a graph.
    ranks_op: a `graphs.GraphsTuple` which contains the ranks of the list
      elements as a graph.
  """
  inputs_op, sort_indices_op, ranks_op = create_graph_dicts_pt(
      batch_size, num_elements_min_max)
  inputs_op = utils_pt.data_dicts_to_graphs_tuple(inputs_op)
  sort_indices_op = utils_pt.data_dicts_to_graphs_tuple(sort_indices_op)
  ranks_op = utils_pt.data_dicts_to_graphs_tuple(ranks_op)

  inputs_op = utils_pt.fully_connect_graph_dynamic(inputs_op)
  sort_indices_op = utils_pt.fully_connect_graph_dynamic(sort_indices_op)
  ranks_op = utils_pt.fully_connect_graph_dynamic(ranks_op)

  targets_op = create_linked_list_target(batch_size, sort_indices_op)
  nodes = torch.cat((targets_op.nodes, 1.0 - targets_op.nodes), axis=1)
  edges = torch.cat((targets_op.edges, 1.0 - targets_op.edges), axis=1)
  targets_op = targets_op._replace(nodes=nodes, edges=edges)

  return inputs_op, targets_op, sort_indices_op, ranks_op


def create_linked_list_target(batch_size, input_graphs):
    """Creates linked list targets.
    
    Returns a graph with the same number of nodes as `input_graph`. Each node
    contains a 2d vector with targets for a 1-class classification where only one
    node is `True`, the smallest value in the array. The vector contains two
    values: [prob_true, prob_false].
    It also contains edges connecting all nodes. These are again 2d vectors with
    softmax targets [prob_true, prob_false]. An edge is True
    if n+1 is the element immediately after n in the sorted list.
    
    Args:
      batch_size: batch size for the `input_graphs`.
      input_graphs: a `graphs.GraphsTuple` which contains a batch of inputs.
    
    Returns:
      A `graphs.GraphsTuple` with the targets, which encode the linked list.
    """
    target_graphs = []
    for i in range(batch_size):
        input_graph = utils_pt.get_graph(input_graphs, i)
        
        num_elements=input_graph.nodes.shape[0]
        
        si = input_graph.nodes.squeeze().type(torch.int32)
        
        if si[:1].item() > num_elements:
            _,idx=si.unique(return_inverse=True)
            tmp = torch.nn.functional.one_hot(idx[:1],num_elements)    
            tmp = torch.zeros_like(tmp)
            nodes = torch.reshape(tmp, (-1,1)).float()
            
        else:    
            tmp = torch.nn.functional.one_hot(si[:1].long(),num_elements)
            nodes = torch.reshape(tmp, (-1,1)).float()   
            
        x = torch.stack((si[:-1], si[1:]))[None]
        y = torch.stack((input_graph.senders, input_graph.receivers), axis=1)[:, :, None]
        
        edges=torch.reshape(torch.any(torch.all(torch.eq(x,y),dim=1), dim=1).float(), (-1,1))
        
        target_graphs.append(input_graph._replace(nodes=nodes, edges=edges))
    return utils_pt.concat(target_graphs, axis=0)

def plot_linked_list(ax, graph, sort_indices):
  """Plot a networkx graph containing weights for the linked list probability."""
  nd = len(graph.nodes())
  probs = np.zeros((nd, nd))
  for edge in graph.edges(data=True):
    probs[edge[0], edge[1]] = edge[2]["features"][0]
  ax.matshow(probs[sort_indices][:, sort_indices], cmap="viridis")
  ax.grid(False)
  
def compute_accuracy(target, output):
  """Calculate model accuracy.

  Returns the number of correctly predicted links and the number
  of completely solved list sorts (100% correct predictions).

  Args:
    target: A `graphs.GraphsTuple` that contains the target graph.
    output: A `graphs.GraphsTuple` that contains the output graph.

  Returns:
    correct: A `float` fraction of correctly labeled nodes/edges.
    solved: A `float` fraction of graphs that are completely correctly labeled.
  """
  tdds = utils_np.graphs_tuple_to_data_dicts(target)
  odds = utils_np.graphs_tuple_to_data_dicts(output)
  cs = []
  ss = []
  for td, od in zip(tdds, odds):
    num_elements = td["nodes"].shape[0]
    xn = np.argmax(td["nodes"], axis=-1)
    yn = np.argmax(od["nodes"], axis=-1)

    xe = np.reshape(
        np.argmax(
            np.reshape(td["edges"], (num_elements, num_elements, 2)), axis=-1),
        (-1,))
    ye = np.reshape(
        np.argmax(
            np.reshape(od["edges"], (num_elements, num_elements, 2)), axis=-1),
        (-1,))
    c = np.concatenate((xn == yn, xe == ye), axis=0)
    s = np.all(c)
    cs.append(c)
    ss.append(s)
  correct = np.mean(np.concatenate(cs, axis=0))
  solved = np.mean(np.stack(ss))
  return correct, solved  
  

####
# num_elements_min_max = (5, 10)

# inputs_op, targets_op, sort_indices_op, ranks_op = create_data_ops(
#     1, num_elements_min_max)

# inputs_nodes, sort_indices_nodes, ranks_nodes, targets  = inputs_op.nodes, sort_indices_op.nodes, ranks_op.nodes, targets_op

# sort_indices = np.squeeze(sort_indices_nodes).type(torch.int64)


# # Plot sort linked lists.
# # The matrix plots show each element from the sorted list (rows), and which
# # element they link to as next largest (columns). Ground truth is a diagonal
# # offset toward the upper-right by one.
# fig = plt.figure(1, figsize=(4, 4))
# fig.clf()
# ax = fig.add_subplot(1, 1, 1)
# plot_linked_list(ax,
#                  utils_np.graphs_tuple_to_networkxs(targets)[0], sort_indices)
# ax.set_title("Element-to-element links for sorted elements")
# ax.set_axis_off()

# fig = plt.figure(2, figsize=(10, 2))
# fig.clf()
# ax1 = fig.add_subplot(1, 3, 1)
# ax2 = fig.add_subplot(1, 3, 2)
# ax3 = fig.add_subplot(1, 3, 3)

# i = 0
# num_elements = ranks_nodes.shape[0]
# inputs = np.squeeze(inputs_nodes)
# ranks = np.squeeze(ranks_nodes * (num_elements - 1.0)).type(torch.int64)
# x = np.arange(inputs.shape[0])

# ax1.set_title("Inputs")
# ax1.barh(x, inputs, color="b")
# ax1.set_xlim(-0.01, 1.01)

# ax2.set_title("Sorted")
# ax2.barh(x, inputs[sort_indices], color="k")
# ax2.set_xlim(-0.01, 1.01)

# ax3.set_title("Ranks")
# ax3.barh(x, ranks, color="r")
# _ = ax3.set_xlim(0, len(ranks) + 0.5)

