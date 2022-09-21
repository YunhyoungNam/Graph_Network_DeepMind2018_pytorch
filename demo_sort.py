# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 17:07:10 2022

@author: darkstar0983
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import time

from src import utils_np
from src import utils_pt
from src import utils
import models
import matplotlib.pyplot as plt
import numpy as np
import torch
import sort

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
        self.co_global_model_info = {'dim' : 16,
                      'n_blk' : 2,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : 64, 
                      'out_dim' : 16, ##_____here________|
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
                      'n_blk' : 2,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : utils.dim_cal_GI(16)['node_dim'], 
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


# Model parameters.
# Number of processing (message-passing) steps.
num_processing_steps_tr = 10
num_processing_steps_ge = 10

# Data / training parameters.
num_training_iterations = 10000
batch_size_tr = 32
batch_size_ge = 100
# Number of elements in each list is sampled uniformly from this range.
num_elements_min_max_tr = (8, 17)
num_elements_min_max_ge = (16,23) #(16, 33)



# Data.
# Training.
inputs_op_tr, targets_op_tr, sort_indices_op_tr, _ = sort.create_data_ops(
    batch_size_tr, num_elements_min_max_tr)
inputs_op_tr = utils_pt.set_zero_edge_features(inputs_op_tr, 1)
inputs_op_tr = utils_pt.set_zero_global_features(inputs_op_tr, 1)
# Test/generalization.
inputs_op_ge, targets_op_ge, sort_indices_op_ge, _ = sort.create_data_ops(
    batch_size_ge, num_elements_min_max_ge)
inputs_op_ge = utils_pt.set_zero_edge_features(inputs_op_ge, 1)
inputs_op_ge = utils_pt.set_zero_global_features(inputs_op_ge, 1)


# network options
opts=network_opts(inputs_op_tr,'sum')
opts.total_iters = num_training_iterations

# Connect the data to the model.
# Instantiate the model.
model = models.EncodeProcessDecode(opts).cuda()

# Loss.

CEloss = torch.nn.CrossEntropyLoss().cuda()
BCEloss = torch.nn.BCEWithLogitsLoss().cuda()

def create_loss_ops(target_op, output_ops, loss_):
  if not isinstance(output_ops, collections.Sequence):
    output_ops = [output_ops]    
  loss_ops = [
      loss_(output_op.nodes, torch.argmax(target_op.nodes,dim=1)) +
      loss_(output_op.edges, torch.argmax(target_op.edges,dim=1))
      for output_op in output_ops
  ]
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
for iteration in range(last_iteration, opts.total_iters):
    
    # inputs_op_tr, targets_op_tr, sort_indices_op_tr, _ = sort.create_data_ops(
    #      batch_size_tr, num_elements_min_max_tr)
    # inputs_op_tr = utils_pt.set_zero_edge_features(inputs_op_tr, 1)
    # inputs_op_tr = utils_pt.set_zero_global_features(inputs_op_tr, 1)    
    
    # cuda appl    
    inputs_op_tr = inputs_op_tr.replace(edges=inputs_op_tr.edges.cuda(),
                                      nodes=inputs_op_tr.nodes.cuda(),
                                      receivers=inputs_op_tr.receivers.cuda(),
                                      senders=inputs_op_tr.senders.cuda(),
                                      n_node=inputs_op_tr.n_node.cuda(),
                                      n_edge=inputs_op_tr.n_edge.cuda())
    
    targets_op_tr = targets_op_tr.replace(edges=targets_op_tr.edges.cuda(),
                                      nodes=targets_op_tr.nodes.cuda(),
                                      receivers=targets_op_tr.receivers.cuda(),
                                      senders=targets_op_tr.senders.cuda(),
                                      n_node=targets_op_tr.n_node.cuda(),
                                      n_edge=targets_op_tr.n_edge.cuda())
    
    optimizer.zero_grad()
    
    output_ops_tr = model(inputs_op_tr, num_processing_steps_tr)
    
    # Training loss.
    loss_ops_tr = create_loss_ops(targets_op_tr, output_ops_tr, CEloss)

    # Loss across processing steps.
    loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr
    
    loss_op_tr.backward()
    optimizer.step()
    
    scheduler.step()
    
    the_time = time.time()
    elapsed_since_last_log = the_time - last_log_time    
    
    if elapsed_since_last_log > log_every_seconds:        
        model.eval() 

        # inputs_op_ge, targets_op_ge, sort_indices_op_ge, _ = sort.create_data_ops(
        #     batch_size_ge, num_elements_min_max_ge)
        # inputs_op_ge = utils_pt.set_zero_edge_features(inputs_op_ge, 1)
        # inputs_op_ge = utils_pt.set_zero_global_features(inputs_op_ge, 1)    
  
        # cuda appl    
        inputs_op_ge = inputs_op_ge.replace(edges=inputs_op_ge.edges.cuda(),
                                            nodes=inputs_op_ge.nodes.cuda(),
                                            receivers=inputs_op_ge.receivers.cuda(),
                                            senders=inputs_op_ge.senders.cuda(),
                                            n_edge=inputs_op_ge.n_edge.cuda(),
                                            n_node=inputs_op_ge.n_node.cuda())
        targets_op_ge = targets_op_ge.replace(edges=targets_op_ge.edges.cuda(),
                                            nodes=targets_op_ge.nodes.cuda(),
                                            receivers=targets_op_ge.receivers.cuda(),
                                            senders=targets_op_ge.senders.cuda(),
                                            n_edge=targets_op_ge.n_edge.cuda(),
                                            n_node=targets_op_ge.n_node.cuda())        
        
        output_ops_ge = model(inputs_op_ge, num_processing_steps_ge)   
        
        loss_ops_ge = create_loss_ops(targets_op_ge, output_ops_ge, CEloss)
        loss_op_ge = loss_ops_ge[-1]  # Loss from final processing step.        
        
        model.train()    
    
    
        targets_op_tr_ = targets_op_tr.replace(edges=targets_op_tr.edges.detach().cpu().numpy(),
                                      nodes=targets_op_tr.nodes.detach().cpu().numpy(),
                                      receivers=targets_op_tr.receivers.detach().cpu().numpy(),
                                      senders=targets_op_tr.senders.detach().cpu().numpy(),
                                      n_node=targets_op_tr.n_node.detach().cpu().numpy(),
                                      n_edge=targets_op_tr.n_edge.detach().cpu().numpy())
        
        targets_op_ge_ = targets_op_ge.replace(edges=targets_op_ge.edges.detach().cpu().numpy(),
                                      nodes=targets_op_ge.nodes.detach().cpu().numpy(),
                                      receivers=targets_op_ge.receivers.detach().cpu().numpy(),
                                      senders=targets_op_ge.senders.detach().cpu().numpy(),
                                      n_node=targets_op_ge.n_node.detach().cpu().numpy(),
                                      n_edge=targets_op_ge.n_edge.detach().cpu().numpy())
        
    
    
        correct_tr, solved_tr = sort.compute_accuracy(targets_op_tr_, utils_pt.graphs_tuple_pt_to_graphs_tuple(utils_pt.graphs_to_cpu(output_ops_tr[-1])))
        correct_ge, solved_ge = sort.compute_accuracy(targets_op_ge_, utils_pt.graphs_tuple_pt_to_graphs_tuple(utils_pt.graphs_to_cpu(output_ops_ge[-1])))
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
    
    

############
############
#@title Visualize results  { form-width: "30%" }

# This cell visualizes the results of training. You can visualize the
# intermediate results by interrupting execution of the cell above, and running
# this cell. You can then resume training by simply executing the above cell
# again.

# Plot results curves.
fig = plt.figure(11, figsize=(18, 3))
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

# Plot sort linked lists for test/generalization.
# The matrix plots show each element from the sorted list (rows), and which
# element they link to as next largest (columns). Ground truth is a diagonal
# offset toward the upper-right by one.
inputs_op_tr = inputs_op_tr.replace(edges=inputs_op_tr.edges.detach().cpu().numpy(),
                                  nodes=inputs_op_tr.nodes.detach().cpu().numpy(),
                                  receivers=inputs_op_tr.receivers.detach().cpu().numpy(),
                                  senders=inputs_op_tr.senders.detach().cpu().numpy(),
                                  n_node=inputs_op_tr.n_node.detach().cpu().numpy(),
                                  n_edge=inputs_op_tr.n_edge.detach().cpu().numpy())


outputs = utils_np.graphs_tuple_to_networkxs( utils_pt.graphs_tuple_pt_to_graphs_tuple(utils_pt.graphs_to_cpu(output_ops_tr[-1])))
targets = utils_np.graphs_tuple_to_networkxs(targets_op_tr_)
inputs = utils_np.graphs_tuple_to_networkxs(inputs_op_tr)
batch_element = 0
fig = plt.figure(12, figsize=(8, 4.5))
fig.clf()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
sort_indices = np.squeeze(
    utils_np.get_graph(sort_indices_op_tr,
                       batch_element).nodes).astype(int)
fig.suptitle("Element-to-element link predictions for sorted elements")
sort.plot_linked_list(ax1, targets[batch_element], sort_indices)
ax1.set_title("Ground truth")
ax1.set_axis_off()
sort.plot_linked_list(ax2, outputs[batch_element], sort_indices)
ax2.set_title("Predicted")
ax2.set_axis_off()