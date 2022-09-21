# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:28:30 2022

@author: darkstar0983
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from src import blocks
from src import utils_pt
from src import utils
import models
import physics

from matplotlib import pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset

try:
    import seaborn as sns
except ImportError:
    pass
else:
    sns.reset_orig()
 
'''
network architecture information
model_info ={ 'dim' : 32,
              'n_blk' : 4,
              'norm' : 'bn',
              'activ' : 'lrelu',
              'in_dim' : 'none',
              'out_dim' : 'none',
              'norm_final' : 'none',
              'activ_final' : 'none'
              'init_weight' : 'none'                        
              }  
'''

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
        self.en_edge_model_info = {'dim' : 32,
                      'n_blk' : 4,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : 'none',   
                      'out_dim' : 16,      
                      'norm_final' : 'ln',
                      'activ_final' : 'relu',
                      'init_weight' : 'normal'}  
        self.en_node_model_info = {'dim' : 32,
                      'n_blk' : 4,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : 'none', 
                      'out_dim' : 16,
                      'norm_final' : 'ln',
                      'activ_final' : 'relu',
                      'init_weight' : 'normal'}
        self.en_global_model_info = {'dim' : 32,
                      'n_blk' : 4,
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
        self.co_edge_model_info ={'dim' : 32,
                      'n_blk' : 4,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : utils.dim_cal_GN(16*2, 16)['edge_dim'], 
                      'out_dim' : 16, ##_____here________|                      
                      'norm_final' : 'none',
                      'activ_final' : 'none',
                      'init_weight' : 'normal'}
        self.co_node_model_info ={'dim' : 32,
                      'n_blk' : 4,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : utils.dim_cal_GN(16*2, 16)['node_dim'], 
                      'out_dim' : 16, ##_____here________|
                      'norm_final' : 'none',
                      'activ_final' : 'none',
                      'init_weight' : 'normal'}
        self.co_global_model_info ={'dim' : 32,
                      'n_blk' : 4,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : utils.dim_cal_GN(16*2, 16)['global_dim'], 
                      'out_dim' : 16, ##______here_______|
                      'norm_final' : 'none',
                      'activ_final' : 'none',
                      'init_weight' : 'normal'}   
        self.cor_reducer = reducer              
        
        ## decoder part
        '''
        utils.dim_cal_GI : previous out_dim each
        out_dim : value you want
        '''        
        self.de_graph = graph
        self.de_edge_model_info = {'dim' : 32,
                      'n_blk' : 4,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : utils.dim_cal_GI(16)['edge_dim'], 
                      'out_dim' : 16,
                      'norm_final' : 'ln',
                      'activ_final' : 'relu',
                      'init_weight' : 'normal'}
        self.de_node_model_info = {'dim' : 32,
                      'n_blk' : 4,
                      'norm' : 'ln',
                      'activ' : 'relu',
                      'in_dim' : utils.dim_cal_GI(16)['node_dim'], 
                      'out_dim' : 16,
                      'norm_final' : 'ln',
                      'activ_final' : 'relu',
                      'init_weight' : 'normal'}
        self.de_global_model_info = {'dim' : 32,
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
        self.tr_edge_model_info = {'dim' : 32,
                      'n_blk' : 1,
                      'norm' : 'none',
                      'activ' : 'none',
                      'in_dim' : utils.dim_cal_GI(16)['edge_dim'], 
                      'out_dim' : 2,
                      'norm_final' : 'none',
                      'activ_final' : 'none',
                      'init_weight' : 'normal'}
        self.tr_node_model_info = {'dim' : 32,
                      'n_blk' : 1,
                      'norm' : 'none',
                      'activ' : 'none',
                      'in_dim' : utils.dim_cal_GI(16)['node_dim'], 
                      'out_dim' : 2,
                      'norm_final' : 'none',
                      'activ_final' : 'none',
                      'init_weight' : 'normal'}
        self.tr_global_model_info = {'dim' : 32,
                      'n_blk' : 1,
                      'norm' : 'none',
                      'activ' : 'none',
                      'in_dim' : utils.dim_cal_GI(16)['global_dim'], 
                      'out_dim' : 2,
                      'norm_final' : 'none',
                      'activ_final' : 'none',
                      'init_weight' : 'normal'} 
        self.tr_reducer = reducer  


        #####
        self.total_iters = 100000 # num_training_iterations
        self.epoch_count = 0 # the starting epoch count
        self.n_epochs = int(self.total_iters * 0.8) # number of epochs with the initial learning rate
        self.n_epochs_decay = self.total_iters - self.n_epochs
        self.lr_policy = 'linear'


class dataset(Dataset):
    def __init__(self, Simulator,
                 Graph,
                 Num_time_steps,
                 Step_size,
                 Node_noise_level,
                 Edge_noise_level,
                 Global_noise_level):
    
        self.graph = Graph
        self.num_time_steps = Num_time_steps 
        self.step_size = Step_size
        self.node_noise_level = Node_noise_level
        self.edge_noise_level = Edge_noise_level
        self.global_noise_level = Global_noise_level
        self.simulator = simulator
        self.initial_conditions_tr, self.true_trajectory_tr = physics.generate_trajectory(
                                                            Simulator,
                                                            Graph,
                                                            Num_time_steps,
                                                            Step_size,
                                                            Node_noise_level,
                                                            Edge_noise_level,
                                                            Global_noise_level)        
        
    def __getitem__(self, index):
        
        t = index
        input_graph = self.initial_conditions_tr.replace(nodes=self.true_trajectory_tr[t])
        target_nodes = self.true_trajectory_tr[t + 1] 
        
        
        return input_graph, target_nodes
    
    
    def __len__(self):
        return self.true_trajectory_tr.shape[0]   
         

rand = np.random.RandomState(SEED)

# Model parameters.
num_processing_steps_tr = 5
num_processing_steps_ge = 1

# Data / training parameters.
num_training_iterations = 100000
batch_size_tr = 256
batch_size_ge = 100
num_time_steps = 50
step_size = 0.1
num_masses_min_max_tr = (4, 9)
dist_between_masses_min_max_tr = (0.2, 1.0)

# Data.
# Base graphs for training.
num_masses_tr = rand.randint(*num_masses_min_max_tr, size=batch_size_tr)
dist_between_masses_tr = rand.uniform(
    *dist_between_masses_min_max_tr, size=batch_size_tr)
static_graph_tr = [
    physics.base_graph(n, d) for n, d in zip(num_masses_tr, dist_between_masses_tr)
]
base_graph_tr = utils_pt.data_dicts_to_graphs_tuple(static_graph_tr)

# network options
opts=network_opts(base_graph_tr,'sum')

# Create the model.
model = models.EncodeProcessDecode(opts).cuda()

# True physics simulator for data generation.
simulator = physics.SpringMassSimulator(step_size=step_size)

# Training.
# Generate a training trajectory by adding noise to initial
# position, spring constants and gravity
#initial_conditions_tr, true_trajectory_tr = physics.generate_trajectory(
#    simulator,
#    base_graph_tr,
#    num_time_steps,
#    step_size,
#    node_noise_level=0.04,
#    edge_noise_level=5.0,
#    global_noise_level=1.0)

 #################

tdataset=dataset(Simulator=simulator,
                 Graph = base_graph_tr,
                 Num_time_steps=num_time_steps,
                 Step_size=step_size,
                 Node_noise_level=0.04,
                 Edge_noise_level=5.0,
                 Global_noise_level=1.0)


tloader = DataLoader(tdataset, batch_size=1, shuffle=True, num_workers=0)

# Base graphs for testing.
# 4 masses 1m apart in a chain like structure.
base_graph_4_ge = utils_pt.data_dicts_to_graphs_tuple(
    [physics.base_graph(4, 0.5)] * batch_size_ge)

initial_conditions_4_ge, true_trajectory_4_ge = physics.generate_trajectory(
    simulator,
    base_graph_4_ge,
    num_time_steps,
    step_size,
    node_noise_level=0.04,
    edge_noise_level=5.0,
    global_noise_level=1.0)

input_graph_4_ge = initial_conditions_4_ge.replace(edges = initial_conditions_4_ge.edges.cuda(),
                                     nodes = initial_conditions_4_ge.nodes.cuda(),
                                     globals = initial_conditions_4_ge.globals.cuda(),
                                     senders = initial_conditions_4_ge.senders.cuda(),
                                     receivers = initial_conditions_4_ge.receivers.cuda(),
                                     n_node = initial_conditions_4_ge.n_node.cuda(),
                                     n_edge = initial_conditions_4_ge.n_edge.cuda()
                                     )

# 9 masses 0.5m apart in a chain like structure.
base_graph_9_ge = utils_pt.data_dicts_to_graphs_tuple(
    [physics.base_graph(9, 0.5)] * batch_size_ge)

initial_conditions_9_ge, true_trajectory_9_ge = physics.generate_trajectory(
    simulator,
    base_graph_9_ge,
    num_time_steps,
    step_size,
    node_noise_level=0.04,
    edge_noise_level=5.0,
    global_noise_level=1.0)

input_graph_9_ge = initial_conditions_9_ge.replace(edges = initial_conditions_9_ge.edges.cuda(),
                                     nodes = initial_conditions_9_ge.nodes.cuda(),
                                     globals = initial_conditions_9_ge.globals.cuda(),
                                     senders = initial_conditions_9_ge.senders.cuda(),
                                     receivers = initial_conditions_9_ge.receivers.cuda(),
                                     n_node = initial_conditions_9_ge.n_node.cuda(),
                                     n_edge = initial_conditions_9_ge.n_edge.cuda()
                                     )


#####
L2loss = torch.nn.MSELoss()


def loss_cal_train(target, preds, loss_):
    losses = [loss_(pred.nodes[:,0],target[:,2:4][:,0])+loss_(pred.nodes[:,1],target[:,2:4][:,1]) for pred in preds]
    
    return losses
    
def loss_cal_test(targets, preds, loss_):
    num , _, _ = targets.shape
    losses = 0 
    for i in range(num):        
        losses += loss_(preds[i,:,2:4][:,0],targets[i,:,2:4][:,0]) + loss_(preds[i,:,2:4][:,1],targets[i,:,2:4][:,1])
    return losses/(i+1)

'''
def loss_cal_train(target, preds):
    losses = [torch.mean(torch.sum((pred.nodes-target[:,2:4])**2,dim=0)) for pred in preds]
    
    return losses

def loss_cal_test(targets, preds):
    num , _, _ = targets.shape
    losses = 0 
    for i in range(num):
        losses += torch.mean(torch.sum((preds[i,:,2:4]-targets[i,:,2:4])**2,dim=0))
    return losses/(i+1)
'''
lr = 0.0001

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = utils.get_scheduler(optimizer, opts) 

epoch = 0
losses = []
test_losses_4_ge = []
test_losses_9_ge = []

train_loader = iter(tloader)
model.train()

for iters in range(num_training_iterations):   
    
    
    if iters % len(tloader) == 0 and iters != 0:
        epoch += 1
        
    try:
        input_graph_tr, target_nodes_tr = next(train_loader)
    except:
        train_loader = iter(tloader)
        input_graph_tr, target_nodes_tr = next(train_loader)
    #t = np.random.randint(num_time_steps-1)
    #input_graph_tr = initial_conditions_tr.replace(nodes=true_trajectory_tr[t])
    #target_nodes_tr = true_trajectory_tr[t + 1]           
       
    input_graph = input_graph_tr.replace(edges = input_graph_tr.edges.squeeze(0).cuda(),
                                         nodes = input_graph_tr.nodes.squeeze(0).cuda(),
                                         globals = input_graph_tr.globals.squeeze(0).cuda(),
                                         senders = input_graph_tr.senders.squeeze(0).cuda(),
                                         receivers = input_graph_tr.receivers.squeeze(0).cuda(),
                                         n_node = input_graph_tr.n_node.squeeze(0).cuda(),
                                         n_edge = input_graph_tr.n_edge.squeeze(0).cuda()
                                         )
    target_nodes = target_nodes_tr.squeeze(0).cuda()
    
    optimizer.zero_grad()
    
    preds = model(input_graph, num_processing_steps_tr)
    
    loss_list=loss_cal_train(target_nodes, preds, L2loss)
    
    loss = 0
    for value in loss_list:
        loss += value
    loss = loss / num_processing_steps_tr   
    
    loss.backward()
    optimizer.step()
    
    if iters % len(tloader) == 0 and iters != 0:
        scheduler.step()
    
    
    if epoch !=0 and epoch % 5 ==0 and iters % len(tloader) == 0:
        losses.append(loss.data.item())
        print('epoch : {}'.format(epoch))
        print('loss_mean_per_epoch : {:.4f}'.format(np.array(losses).mean()))
        
    if epoch != 0 and epoch % 20 ==0 and iters % len(tloader) == 0:
        print('!!!TEST!!')
        print('epoch : {}, current_lr : {:.6f}'.format(epoch, optimizer.param_groups[0]['lr']))        
        model.eval()
        ##4 masses 1m apart in a chain                      
        _, predicted_nodes_rollout_4_ge = physics.roll_out_physics_model(
            model, input_graph_4_ge, num_processing_steps_tr,
            num_time_steps, step_size)      
        
        test_loss_4_ge = loss_cal_test(true_trajectory_4_ge.cuda(), predicted_nodes_rollout_4_ge, L2loss).data.item()   
        test_losses_4_ge.append(test_loss_4_ge)
        
        ##9 masses 0.5m apart in a chain
        _, predicted_nodes_rollout_9_ge = physics.roll_out_physics_model(
            model, input_graph_9_ge, num_processing_steps_tr,
            num_time_steps, step_size)        
        
        test_loss_9_ge = loss_cal_test(true_trajectory_9_ge.cuda(), predicted_nodes_rollout_9_ge, L2loss).data.item()
        test_losses_9_ge.append(test_loss_9_ge)
        model.train()
        print('####Test Loss####')
        print('4_ge : {:.4f}'.format(test_loss_4_ge))
        print('9_ge : {:.4f}'.format(test_loss_9_ge))
        print('#################')
        
      
'''

### viz
def get_node_trajectories(rollout_array, batch_size):  # pylint: disable=redefined-outer-name
  return np.split(rollout_array[..., :2], batch_size, axis=1)

true_rollouts_4 = get_node_trajectories(true_trajectory_4_ge.data.numpy(),
                                        batch_size_ge)
predicted_rollouts_4 = get_node_trajectories(predicted_nodes_rollout_4_ge.data.cpu().numpy(),
                                             batch_size_ge)       
        
true_rollouts = true_rollouts_4
predicted_rollouts = predicted_rollouts_4


num_graphs = len(true_rollouts)
num_time_steps = true_rollouts[0].shape[0]


# Plot state sequences.
max_graphs_to_plot = 1
num_graphs_to_plot = min(num_graphs, max_graphs_to_plot)
num_steps_to_plot = 24
max_time_step = num_time_steps - 1
step_indices = np.floor(np.linspace(0, max_time_step,
                                    num_steps_to_plot)).astype(int).tolist()
w = 6
h = int(np.ceil(num_steps_to_plot / w))
fig = plt.figure(101, figsize=(18, 8))
fig.clf()
for i, (true_rollout, predicted_rollout) in enumerate(
    zip(true_rollouts, predicted_rollouts)):
  xys = np.hstack([predicted_rollout, true_rollout]).reshape([-1, 2])
  xs = xys[:, 0]
  ys = xys[:, 1]
  b = 0.05
  xmin = xs.min() - b * xs.ptp()
  xmax = xs.max() + b * xs.ptp()
  ymin = ys.min() - b * ys.ptp()
  ymax = ys.max() + b * ys.ptp()
  if i >= num_graphs_to_plot:
    break
  for j, step_index in enumerate(step_indices):
    iax = i * w + j + 1
    ax = fig.add_subplot(h, w, iax)
    ax.plot(
        true_rollout[step_index, :, 0],
        true_rollout[step_index, :, 1],
        "k",
        label="True")
    ax.plot(
        predicted_rollout[step_index, :, 0],
        predicted_rollout[step_index, :, 1],
        "r",
        label="Predicted")
    ax.set_title("Example {:02d}: frame {:03d}".format(i, step_index))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    if j == 0:
      ax.legend(loc=3)



# Plot x and y trajectories over time.
max_graphs_to_plot = 3
num_graphs_to_plot = min(len(true_rollouts), max_graphs_to_plot)
w = 2
h = num_graphs_to_plot
fig = plt.figure(102, figsize=(18, 12))
fig.clf()
for i, (true_rollout, predicted_rollout) in enumerate(
    zip(true_rollouts, predicted_rollouts)):
  if i >= num_graphs_to_plot:
    break
  t = np.arange(num_time_steps)
  for j in range(2):
    coord_string = "x" if j == 0 else "y"
    iax = i * 2 + j + 1
    ax = fig.add_subplot(h, w, iax)
    ax.plot(t, true_rollout[..., j], "k", label="True")
    ax.plot(t, predicted_rollout[..., j], "r", label="Predicted")
    ax.set_xlabel("Time")
    ax.set_ylabel("{} coordinate".format(coord_string))
    ax.set_title("Example {:02d}: Predicted vs actual coords over time".format(
        i))
    ax.set_frame_on(False)
    if i == 0 and j == 1:
      handles, labels = ax.get_legend_handles_labels()
      unique_labels = []
      unique_handles = []
      for i, (handle, label) in enumerate(zip(handles, labels)):  # pylint: disable=redefined-outer-name
        if label not in unique_labels:
          unique_labels.append(label)
          unique_handles.append(handle)
      ax.legend(unique_handles, unique_labels, loc=3)
'''      