# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:19:56 2022

@author: darkstar0983
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import time

from src import graphs
from src import utils_np
from src import utils_pt
from src import utils
import models
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import spatial
import torch

import shortestpath

###### newtork information  options
class network_opts():
    def __init__(self, graph, reducer):
        super(network_opts, self).__init__()
        '''
        grpah is the GraphTuple
        '''
        
        ## encoder part
        '''
        in_dim is automatically determined by initial graph information : fixed as 'none'
        out_dim depends on your selection
        '''
        self.en_graph = graph
        self.en_edge_model_info = {'dim' : 16,
                      'n_blk' : 2,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : 'none',   
                      'out_dim' : 16,      
                      'norm_final' : 'ln',
                      'activ_final' : 'relu',
                      'init_weight' : 'normal'}  
        self.en_node_model_info = {'dim' : 16,
                      'n_blk' : 2,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : 'none', 
                      'out_dim' : 16,
                      'norm_final' : 'ln',
                      'activ_final' : 'relu',
                      'init_weight' : 'normal'}
        self.en_global_model_info = {'dim' : 16,
                      'n_blk' : 2,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : 'none', 
                      'out_dim' : 16,
                      'norm_final' : 'ln',
                      'activ_final' : 'relu',
                      'init_weight' : 'normal'} 
        self.en_reducer = reducer           
        
        ## core part
        '''
        edge out_dim  :  value you want
        utils.dim_cal_GN(previous out_dim, edge out_dim)
        input dim of each network is correlated

        '''
        self.co_graph = graph
        self.co_edge_model_info ={'dim' : 16,
                      'n_blk' : 2,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : utils.dim_cal_GN(16*2, 16)['edge_dim'], 
                      'out_dim' : 16, ##_____here________|                      
                      'norm_final' : 'ln',
                      'activ_final' : 'relu',
                      'init_weight' : 'normal'}
        self.co_node_model_info ={'dim' : 16,
                      'n_blk' : 2,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : utils.dim_cal_GN(16*2, 16)['node_dim'], 
                      'out_dim' : 16, ##_____here________|
                      'norm_final' : 'ln',
                      'activ_final' : 'relu',
                      'init_weight' : 'normal'}
        self.co_global_model_info ={'dim' : 16,
                      'n_blk' : 4,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : utils.dim_cal_GN(16*2, 16)['global_dim'], 
                      'out_dim' : 16, ##______here_______|
                      'norm_final' : 'ln',
                      'activ_final' : 'relu',
                      'init_weight' : 'normal'}   
        self.cor_reducer = reducer              
        
        ## decoder part
        '''
        utils.dim_cal_GI : previous out_dim each
        out_dim : value you want
        '''        
        self.de_graph = graph
        self.de_edge_model_info = {'dim' : 16,
                      'n_blk' : 2,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : utils.dim_cal_GI(16)['edge_dim'], 
                      'out_dim' : 16,
                      'norm_final' : 'ln',
                      'activ_final' : 'relu',
                      'init_weight' : 'normal'}
        self.de_node_model_info = {'dim' : 16,
                      'n_blk' : 2,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : utils.dim_cal_GI(16)['node_dim'], 
                      'out_dim' : 16,
                      'norm_final' : 'ln',
                      'activ_final' : 'relu',
                      'init_weight' : 'normal'}
        self.de_global_model_info = {'dim' : 16,
                      'n_blk' : 4,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : utils.dim_cal_GI(16)['global_dim'], 
                      'out_dim' : 16,
                      'norm_final' : 'ln',
                      'activ_final' : 'relu',
                      'init_weight' : 'normal'}
        self.de_reducer = reducer

        ## output transform part
        '''
        utils.dim_cal_GI : previous out_dim each
        out_dim : value you want
        '''        
        self.tr_graph = graph
        self.tr_edge_model_info = {'dim' : 16,
                      'n_blk' : 1,
                      'norm' : 'none',
                      'activ' : 'none',
                      'in_dim' : utils.dim_cal_GI(16)['edge_dim'], 
                      'out_dim' : 2,
                      'norm_final' : 'none',
                      'activ_final' : 'none',
                      'init_weight' : 'normal'}
        self.tr_node_model_info = {'dim' : 16,
                      'n_blk' : 1,
                      'norm' : 'none',
                      'activ' : 'none',
                      'in_dim' : utils.dim_cal_GI(16)['node_dim'], 
                      'out_dim' : 2,
                      'norm_final' : 'none',
                      'activ_final' : 'none',
                      'init_weight' : 'normal'}
        self.tr_global_model_info = None
        self.tr_reducer = reducer  


        #####
        self.total_iters = 10000 # num_training_iterations
        self.epoch_count = 0 # the starting epoch count
        self.n_epochs = int(self.total_iters * 0.8) # number of epochs with the initial learning rate
        self.n_epochs_decay = self.total_iters - self.n_epochs
        self.lr_policy = 'linear'


###############################################################################
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
rand = np.random.RandomState(seed=SEED)

num_examples = 15  #@param{type: 'integer'}
# Large values (1000+) make trees. Try 20-60 for good non-trees.
theta = 20  #@param{type: 'integer'}
num_nodes_min_max = (16, 17)

input_graphs, target_graphs, graphs_ = shortestpath.generate_networkx_graphs(
    rand, num_examples, num_nodes_min_max, theta)

'''
num = min(num_examples, 16)
w = 3
h = int(np.ceil(num / w))
fig = plt.figure(40, figsize=(w * 4, h * 4))
fig.clf()
for j, graph in enumerate(graphs_):
  ax = fig.add_subplot(h, w, j + 1)
  pos = shortestpath.get_node_dict(graph, "pos")
  plotter = shortestpath.GraphPlotter(ax, graph, pos)
  plotter.draw_graph_with_solution()
'''



# Model parameters.
# Number of processing (message-passing) steps.
num_processing_steps_tr = 10
num_processing_steps_ge = 10

# Data / training parameters.
num_training_iterations = 10000
theta = 20  # Large values (1000+) make trees. Try 20-60 for good non-trees.
batch_size_tr = 32
batch_size_ge = 100
# Number of nodes per graph sampled uniformly from this range.
num_nodes_min_max_tr = (8, 17)
num_nodes_min_max_ge = (16, 33)

input_ph, _, _ = shortestpath.create_data(rand, batch_size_tr, num_nodes_min_max_tr, theta)


# network options
opts=network_opts(input_ph,'sum')

# Create the model.
model = models.EncodeProcessDecode(opts).cuda()

CEloss = torch.nn.CrossEntropyLoss().cuda()
BCEloss = torch.nn.BCEWithLogitsLoss().cuda()

def create_loss_ops(target_op, output_ops, loss_):  
  loss_ops = [
      loss_(output_op.nodes, torch.argmax(target_op.nodes,dim=1)) +
      loss_(output_op.edges, torch.argmax(target_op.edges,dim=1))
      for output_op in output_ops]

  return loss_ops


lr = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = utils.get_scheduler(optimizer, opts) 

log_every_seconds = 60

last_iteration = 0
logged_iterations = []
losses_tr = []
corrects_tr = []
solveds_tr = []
losses_ge = []
corrects_ge = []
solveds_ge = []

start_time = time.time()
last_log_time = start_time
model.train()
for iteration in range(last_iteration, opts.total_iters):
    
    input_ph, target_ph, _ = shortestpath.create_data(rand, batch_size_tr, num_nodes_min_max_tr, theta)
    
    input_ph=utils_pt.graphs_tuple_to_graphs_tuple_pt(input_ph)
    target_ph=utils_pt.graphs_tuple_to_graphs_tuple_pt(target_ph)
    
    # cuda appl    
    input_ph= utils_pt.graphs_to_cuda(input_ph)
    target_ph= utils_pt.graphs_to_cuda(target_ph)
    
    optimizer.zero_grad()


    output_ops_tr = model(input_ph, num_processing_steps_tr)
    
    # Training loss.
    loss_ops_tr = create_loss_ops(target_ph, output_ops_tr, CEloss) 
    
    # Loss across processing steps.
    loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr
    
    loss_op_tr.backward()
    optimizer.step()
    
    scheduler.step()
    
    the_time = time.time()
    elapsed_since_last_log = the_time - last_log_time    
    
    if elapsed_since_last_log > log_every_seconds:        
        model.eval()
        
        input_ph_, target_ph_, raw_graphs = shortestpath.create_data(rand, batch_size_ge, num_nodes_min_max_ge, theta)
        
        input_ph_=utils_pt.graphs_tuple_to_graphs_tuple_pt(input_ph_)
        target_ph_=utils_pt.graphs_tuple_to_graphs_tuple_pt(target_ph_)        
        
        # cuda appl
        input_ph_ = input_ph_.replace(edges = input_ph_.edges.cuda(),
                                             nodes = input_ph_.nodes.cuda(),
                                             globals = input_ph_.globals.cuda(),
                                             senders = input_ph_.senders.cuda(),
                                             receivers = input_ph_.receivers.cuda(),
                                             n_node = input_ph_.n_node.cuda(),
                                             n_edge = input_ph_.n_edge.cuda()
                                             ) 
        
        target_ph_ = target_ph_.replace(edges = target_ph_.edges.cuda(),
                                             nodes = target_ph_.nodes.cuda(),
                                             globals = target_ph_.globals.cuda(),
                                             senders = target_ph_.senders.cuda(),
                                             receivers = target_ph_.receivers.cuda(),
                                             n_node = target_ph_.n_node.cuda(),
                                             n_edge = target_ph_.n_edge.cuda()
                                             )        
        # cuda appl    
        input_ph_ = utils_pt.graphs_to_cuda(input_ph_)
        target_ph_ = utils_pt.graphs_to_cuda(target_ph_)        
        
        output_ops_ge = model(input_ph_, num_processing_steps_ge)    
        
        loss_ops_ge = create_loss_ops(target_ph_, output_ops_ge, CEloss)
        loss_op_ge = loss_ops_ge[-1]  # Loss from final processing step.        
        
        model.train()
        
        correct_tr, solved_tr = shortestpath.compute_accuracy(
        utils_pt.graphs_tuple_pt_to_graphs_tuple(utils_pt.graphs_to_cpu(target_ph)),
        utils_pt.graphs_tuple_pt_to_graphs_tuple(utils_pt.graphs_to_cpu(output_ops_tr[-1])), use_edges=True)
        correct_ge, solved_ge = shortestpath.compute_accuracy(
        utils_pt.graphs_to_cpu(target_ph_), utils_pt.graphs_to_cpu(output_ops_ge[-1]), use_edges=True)
        elapsed = time.time() - start_time
        losses_tr.append(loss_op_tr.data.cpu().item())
        corrects_tr.append(correct_tr)
        solveds_tr.append(solved_tr)
        losses_ge.append(loss_op_ge.data.cpu().item())
        corrects_ge.append(correct_ge)
        solveds_ge.append(solved_ge)
        logged_iterations.append(iteration)
        print("# {:05d}, T {:.1f}, Ltr {:.4f}, Lge {:.4f}, Ctr {:.4f}, Str"
          " {:.4f}, Cge {:.4f}, Sge {:.4f}".format(
              iteration, elapsed, loss_op_tr.data.cpu().item(), loss_op_ge.data.cpu().item(),
              correct_tr, solved_tr, correct_ge, solved_ge)) 
        
        
        
###

def softmax_prob_last_dim(x):  # pylint: disable=redefined-outer-name
  e = np.exp(x)
  return e[:, -1] / np.sum(e, axis=-1)


# Plot results curves.
fig = plt.figure(1, figsize=(18, 3))
fig.clf()
x = np.array(logged_iterations)
# Loss.
y_tr = losses_tr
y_ge = losses_ge
ax = fig.add_subplot(1, 3, 1)
ax.plot(x, y_tr, "k", label="Training")
ax.plot(x, y_ge, "k--", label="Test/generalization")
ax.set_title("Loss across training")
ax.set_xlabel("Training iteration")
ax.set_ylabel("Loss (binary cross-entropy)")
ax.legend()
# Correct.
y_tr = corrects_tr
y_ge = corrects_ge
ax = fig.add_subplot(1, 3, 2)
ax.plot(x, y_tr, "k", label="Training")
ax.plot(x, y_ge, "k--", label="Test/generalization")
ax.set_title("Fraction correct across training")
ax.set_xlabel("Training iteration")
ax.set_ylabel("Fraction nodes/edges correct")
# Solved.
y_tr = solveds_tr
y_ge = solveds_ge
ax = fig.add_subplot(1, 3, 3)
ax.plot(x, y_tr, "k", label="Training")
ax.plot(x, y_ge, "k--", label="Test/generalization")
ax.set_title("Fraction solved across training")
ax.set_xlabel("Training iteration")
ax.set_ylabel("Fraction examples solved")

# Plot graphs and results after each processing step.
# The white node is the start, and the black is the end. Other nodes are colored
# from red to purple to blue, where red means the model is confident the node is
# off the shortest path, blue means the model is confident the node is on the
# shortest path, and purplish colors mean the model isn't sure.
max_graphs_to_plot = 3#6
num_steps_to_plot = 4
node_size = 120
min_c = 0.3
num_graphs = len(raw_graphs)
targets = utils_np.graphs_tuple_to_data_dicts(utils_pt.graphs_to_cpu(target_ph_))
step_indices = np.floor(
    np.linspace(0, num_processing_steps_ge - 1,
                num_steps_to_plot)).astype(int).tolist()
outputs = list(
    zip(*(utils_np.graphs_tuple_to_data_dicts(utils_pt.graphs_to_cpu(output_ops_ge[i]))
          for i in step_indices)))
h = min(num_graphs, max_graphs_to_plot)
w = num_steps_to_plot + 1
fig = plt.figure(101, figsize=(18, h * 3))
fig.clf()
ncs = []
for j, (graph, target, output) in enumerate(zip(raw_graphs, targets, outputs)):
  if j >= h:
    break
  pos = shortestpath.get_node_dict(graph, "pos")
  ground_truth = target["nodes"][:, -1]
  # Ground truth.
  iax = j * (1 + num_steps_to_plot) + 1
  ax = fig.add_subplot(h, w, iax)
  plotter = shortestpath.GraphPlotter(ax, graph, pos)
  color = {}
  for i, n in enumerate(plotter.nodes):
    color[n] = np.array([1.0 - ground_truth[i], 0.0, ground_truth[i], 1.0
                        ]) * (1.0 - min_c) + min_c
  plotter.draw_graph_with_solution(node_size=node_size, node_color=color)
  ax.set_axis_on()
  ax.set_xticks([])
  ax.set_yticks([])
  try:
    ax.set_facecolor([0.9] * 3 + [1.0])
  except AttributeError:
    ax.set_axis_bgcolor([0.9] * 3 + [1.0])
  ax.grid(None)
  ax.set_title("Ground truth\nSolution length: {}".format(
      plotter.solution_length))
  # Prediction.
  for k, outp in enumerate(output):
    iax = j * (1 + num_steps_to_plot) + 2 + k
    ax = fig.add_subplot(h, w, iax)
    plotter = shortestpath.GraphPlotter(ax, graph, pos)
    color = {}
    prob = softmax_prob_last_dim(outp["nodes"].numpy())
    for i, n in enumerate(plotter.nodes):
      color[n] = np.array([1.0 - prob[n], 0.0, prob[n], 1.0
                          ]) * (1.0 - min_c) + min_c
    plotter.draw_graph_with_solution(node_size=node_size, node_color=color)
    ax.set_title("Model-predicted\nStep {:02d} / {:02d}".format(
        step_indices[k] + 1, step_indices[-1] + 1))     