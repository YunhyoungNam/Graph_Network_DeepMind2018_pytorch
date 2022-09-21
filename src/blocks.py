# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 10:43:39 2022

@author: darkstar0983
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from graph_nets import _base
import graphs
import utils_pt
import utils

import torch 
import torch.nn as nn
import torch.nn.functional as F
from random import randint
from graphs import GraphsTuple
from networks import *

NODES = graphs.NODES
EDGES = graphs.EDGES
GLOBALS = graphs.GLOBALS
RECEIVERS = graphs.RECEIVERS
SENDERS = graphs.SENDERS
GLOBALS = graphs.GLOBALS
N_NODE = graphs.N_NODE
N_EDGE = graphs.N_EDGE

def _validate_graph(graph, mandatory_fields, additional_message=None):
  for field in mandatory_fields:
    if getattr(graph, field) is None:
      message = "`{}` field cannot be None".format(field)
      if additional_message:
        message += " " + format(additional_message)
      message += "."
      raise ValueError(message)     
      
def _validate_broadcasted_graph(graph, from_field, to_field):
  additional_message = "when broadcasting {} to {}".format(from_field, to_field)
  _validate_graph(graph, [from_field, to_field], additional_message)

      
def _get_static_num_nodes(graph):
  """Returns the static total number of nodes in a batch or None."""
  return None if graph.nodes is None else list(graph.nodes.shape)[0]

def _get_static_num_edges(graph):
  """Returns the static total number of edges in a batch or None."""
  return None if graph.senders is None else list(graph.senders.shape)[0]

def broadcast_globals_to_edges(graph):
    _validate_broadcasted_graph(graph, GLOBALS, N_EDGE)
    return utils_pt.repeat(graph.globals, graph.n_edge)

def broadcast_globals_to_nodes(graph):
    _validate_broadcasted_graph(graph, GLOBALS, N_NODE)
    return utils_pt.repeat(graph.globals, graph.n_node)

def broadcast_sender_nodes_to_edges(graph):
    _validate_broadcasted_graph(graph, NODES, SENDERS)
    return graph.nodes.index_select(index=graph.senders.long(),dim=0)
    #return utils_pt.index_select(graph.nodes, graph.senders.long())
    #return torch.gather(graph.nodes, 0, graph.senders)
    
def broadcast_receiver_nodes_to_edges(graph):
    _validate_broadcasted_graph(graph, NODES, RECEIVERS)
    return graph.nodes.index_select(index=graph.receivers.long(),dim=0)
    #return utils_pt.index_select(graph.nodes, graph.receivers.long())
    #return torch.gather(graph.nodes, 0, graph.receivers)

class EdgesToGlobalsAggregator(nn.Module):
    def __init__(self, reducer:str):
        super(EdgesToGlobalsAggregator, self).__init__()
        self._reducer = reducer
        '''
        Examples of compatible reducers are:
            str : "sum" & "mean"
        '''
        
    def forward(self, graph):
        _validate_graph(graph, (EDGES,),
                    additional_message="when aggregating from edges.") 
        
        num_graphs = utils_pt.get_num_graphs(graph)
        if num_graphs.device.type == 'cuda':
            graph_index = torch.arange(num_graphs,dtype=torch.int32).cuda()
        else:
            graph_index = torch.arange(num_graphs,dtype=torch.int32)
        indices = utils_pt.repeat(graph_index, graph.n_edge)
        
        return utils_pt.unsorted_segment_cal(graph.edges, indices.long(), num_graphs, self._reducer)          
        
        
class NodesToGlobalsAggregator(nn.Module):
    def __init__(self, reducer:str):
        super(NodesToGlobalsAggregator, self).__init__()
        self._reducer = reducer
        '''
        Examples of compatible reducers are:
            str : "sum" & "mean"
        '''    
            
    def forward(self, graph):
        _validate_graph(graph, (NODES,),
                        additional_message="when aggregating from nodes.")
        num_graphs = utils_pt.get_num_graphs(graph)
        if num_graphs.device.type == 'cuda':
            graph_index = torch.arange(num_graphs,dtype=torch.int32).cuda()
        else:
            graph_index = torch.arange(num_graphs,dtype=torch.int32)
        indices = utils_pt.repeat(graph_index, graph.n_node) 
        
        return utils_pt.unsorted_segment_cal(graph.nodes, indices.long(), num_graphs, self._reducer)        
        
        
class Aggregator(nn.Module):
    def __init__(self, mode):
        super(Aggregator, self).__init__()
        self.mode = mode
        '''
        mode : receivers ==> ReceivedEdgesToNodesAggregator
        mode : senders ==> SentEdgesToNodesAggregator
        '''

    def forward(self, graph):
        edges = graph.edges
        nodes = graph.nodes
        if self.mode == 'receivers':
            indeces = graph.receivers
        elif self.mode == 'senders':
            indeces = graph.senders
        else:
            raise AttributeError("invalid parameter `mode`")
        N_edges, N_features = edges.shape
        N_nodes=nodes.shape[0]
        aggrated_list = []
        for i in range(N_nodes):
            aggrated = edges[indeces == i]
            if aggrated.shape[0] == 0:
                aggrated = torch.zeros(1, N_features)
            aggrated_list.append(torch.sum(aggrated, dim=0))
        return torch.stack(aggrated_list,dim=0)        

class _EdgesToNodesAggregator(nn.Module):
    def __init__(self, reducer, use_sent_edges=False):
        super(_EdgesToNodesAggregator, self).__init__()
        self._reducer = reducer
        self._use_sent_edges = use_sent_edges
        
    def forward(self, graph):
        _validate_graph(graph, (EDGES, SENDERS, RECEIVERS,),
                        additional_message="when aggregating from edges.")  
        
        #if graph.nodes is not None and list(graph.nodes.shape)[0] is not None:
        #  num_nodes = list(graph.nodes.shape)[0]
        #else:
        num_nodes = torch.sum(graph.n_node,dtype=torch.int32) #tf.reduce_sum(graph.n_node)
        indices = graph.senders if self._use_sent_edges else graph.receivers
        return utils_pt.unsorted_segment_cal(graph.edges, indices.long(), num_nodes, self._reducer)    

class SentEdgesToNodesAggregator(_EdgesToNodesAggregator):
  """Agregates sent edges into the corresponding sender nodes."""

  def __init__(self, reducer):
    """Constructor.
    The reducer is used for combining per-edge features (one set of edge
    feature vectors per node) to give per-node features (one feature
    vector per node). The reducer should take a `Tensor` of edge features, a
    `Tensor` of segment indices, and a number of nodes. It should be invariant
    under permutation of edge features within each segment.
    Examples of compatible reducers are:
    * tf.math.unsorted_segment_sum
    * tf.math.unsorted_segment_mean
    * tf.math.unsorted_segment_prod
    * unsorted_segment_min_or_zero
    * unsorted_segment_max_or_zero
    Args:
      reducer: A function for reducing sets of per-edge features to individual
        per-node features.
      name: The module name.
    """
    super(SentEdgesToNodesAggregator, self).__init__(
        use_sent_edges=True, reducer=reducer)


class ReceivedEdgesToNodesAggregator(_EdgesToNodesAggregator):
  """Agregates received edges into the corresponding receiver nodes."""

  def __init__(self, reducer):
    """Constructor.
    The reducer is used for combining per-edge features (one set of edge
    feature vectors per node) to give per-node features (one feature
    vector per node). The reducer should take a `Tensor` of edge features, a
    `Tensor` of segment indices, and a number of nodes. It should be invariant
    under permutation of edge features within each segment.
    Examples of compatible reducers are:
    * tf.math.unsorted_segment_sum
    * tf.math.unsorted_segment_mean
    * tf.math.unsorted_segment_prod
    * unsorted_segment_min_or_zero
    * unsorted_segment_max_or_zero
    Args:
      reducer: A function for reducing sets of per-edge features to individual
        per-node features.
      name: The module name.
    """
    super(ReceivedEdgesToNodesAggregator, self).__init__(
        use_sent_edges=False, reducer=reducer)

    
        
'''
EdgesToGlobalsAggregator(reducer="sum")(graphs_tuple)
NodesToGlobalsAggregator(reducer="sum")(graphs_tuple)

Aggregator('receivers')(graphs_tuple)
ReceivedEdgesToNodesAggregator(reducer="sum")(graphs_tuple)

Aggregator('senders')(graphs_tuple)
SentEdgesToNodesAggregator(reducer="sum")(graphs_tuple)
'''
        
class EdgeBlock(nn.Module):
    def __init__(self,
                 model_info : dict,
                 graph : GraphsTuple,
                 use_edges=True,
                 use_receiver_nodes=True,
                 use_sender_nodes=True,
                 use_globals=True):
        
        super(EdgeBlock, self).__init__()
        if not (use_edges or use_sender_nodes or use_receiver_nodes or use_globals):
          raise ValueError("At least one of use_edges, use_sender_nodes, "
                           "use_receiver_nodes or use_globals must be True.")
    
        self._use_edges = use_edges
        self._use_receiver_nodes = use_receiver_nodes
        self._use_sender_nodes = use_sender_nodes
        self._use_globals = use_globals  
        
        
        N_features = 0
        if model_info['out_dim'] == 'none':
            pre_features=graph.edges.shape[-1]
        else:
            pre_features= model_info['out_dim']
        if self._use_edges and graph.edges is not None:
            N_features += graph.edges.shape[-1]
        if self._use_receiver_nodes and graph.nodes is not None:
            N_features += graph.nodes.shape[-1]
        if self._use_sender_nodes and graph.nodes is not None:
            N_features += graph.nodes.shape[-1]
        if self._use_globals and graph.globals is not None:
            N_features += graph.globals.shape[-1] 
        if model_info['in_dim'] != 'none':
            N_features = model_info['in_dim']            
        
        self._edge_model = MLP(N_features, pre_features, model_info['dim'],
                               model_info['n_blk'], model_info['norm'], model_info['activ'])
        
        if model_info['init_weight'] != 'none':
            utils.init_weights(self._edge_model,model_info['init_weight'])
        
    def forward(self, graph, edge_model_kwargs=None):
        if edge_model_kwargs is None:
            edge_model_kwargs = {}        
            
        _validate_graph(
            graph, (SENDERS, RECEIVERS, N_EDGE), " when using an EdgeBlock")
    
        edges_to_collect = []       

        if self._use_edges and graph.edges is not None:
            _validate_graph(graph, (EDGES,), "when use_edges == True")
            edges_to_collect.append(graph.edges.cuda())
    
        if self._use_receiver_nodes and graph.nodes is not None:
            edges_to_collect.append(broadcast_receiver_nodes_to_edges(graph).cuda())
    
        if self._use_sender_nodes and graph.nodes is not None:
            edges_to_collect.append(broadcast_sender_nodes_to_edges(graph).cuda())
    
        if self._use_globals and graph.globals is not None:
            #num_edges_hint = _get_static_num_edges(graph)
            edges_to_collect.append(broadcast_globals_to_edges(graph).cuda())
            
            
        collected_edges = torch.cat(edges_to_collect, dim=-1)
        updated_edges = self._edge_model(collected_edges, **edge_model_kwargs)
        return graph.replace(edges=updated_edges)


class NodeBlock(nn.Module):
    def __init__(self,
                 model_info : dict,
                 graph : GraphsTuple,
                 use_received_edges=True,
                 use_sent_edges=False,
                 use_nodes=True,
                 use_globals=True,
                 received_edges_reducer='sum',
                 sent_edges_reducer='sum'): 
        
        super(NodeBlock, self).__init__()       
        if not (use_nodes or use_sent_edges or use_received_edges or use_globals):
          raise ValueError("At least one of use_received_edges, use_sent_edges, "
                           "use_nodes or use_globals must be True.")
            
        self._use_received_edges = use_received_edges
        self._use_sent_edges = use_sent_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals
        
        if self._use_received_edges:
            if received_edges_reducer is None:
                raise ValueError("If `use_received_edges==True`, `received_edges_reducer` "
                  "should not be None.")
            self._received_edges_aggregator = ReceivedEdgesToNodesAggregator(received_edges_reducer)
        if self._use_sent_edges:
            if sent_edges_reducer is None:
                raise ValueError("If `use_sent_edges==True`, `sent_edges_reducer` "
                  "should not be None.")
            self._sent_edges_aggregator = SentEdgesToNodesAggregator(sent_edges_reducer)                   


        N_features = 0
        if model_info['out_dim'] == 'none':
            pre_features=graph.nodes.shape[-1]
        else:
            pre_features= model_info['out_dim']
        if self._use_nodes and graph.nodes is not None:
            N_features += graph.nodes.shape[-1]
        if self._use_received_edges and graph.edges is not None:
            N_features += graph.edges.shape[-1]
        if self._use_sent_edges and graph.edges is not None:
            N_features += graph.edges.shape[-1]
        if self._use_globals and graph.globals is not None:
            N_features += graph.globals.shape[-1]  
        if model_info['in_dim'] != 'none':
            N_features = model_info['in_dim']            
            
        self._node_model =  MLP(N_features, pre_features, model_info['dim'],
                                model_info['n_blk'], model_info['norm'], model_info['activ'])

        if model_info['init_weight'] != 'none':
            utils.init_weights(self._node_model,model_info['init_weight'])        
        
    def forward(self, graph, node_model_kwargs=None):
        if node_model_kwargs is None:
          node_model_kwargs = {}
    
        nodes_to_collect = []
    
        if self._use_received_edges and graph.edges is not None:
          nodes_to_collect.append(self._received_edges_aggregator(graph).cuda())
    
        if self._use_sent_edges and graph.edges is not None:
          nodes_to_collect.append(self._sent_edges_aggregator(graph).cuda())
    
        if self._use_nodes and  graph.nodes is not None:
          _validate_graph(graph, (NODES,), "when use_nodes == True")
          nodes_to_collect.append(graph.nodes.cuda())
    
        if self._use_globals and graph.globals is not None:
          # The hint will be an integer if the graph has node features and the total
          # number of nodes is known at tensorflow graph definition time, or None
          # otherwise.
          #num_nodes_hint = _get_static_num_nodes(graph)
          nodes_to_collect.append(broadcast_globals_to_nodes(graph).cuda())
    
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)            
        updated_nodes = self._node_model(collected_nodes, **node_model_kwargs)
        return graph.replace(nodes=updated_nodes)
        

class GlobalBlock(nn.Module):
    def __init__(self,
                 model_info : dict,
                 graph : GraphsTuple,
                 use_edges=True,
                 use_nodes=True,
                 use_globals=True,
                 nodes_reducer='sum',
                 edges_reducer='sum'):  
        
        super(GlobalBlock, self).__init__()        
        if not (use_nodes or use_edges or use_globals):
            raise ValueError("At least one of use_edges, "
                       "use_nodes or use_globals must be True.")        
        
        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals


        if self._use_edges:
            if edges_reducer is None:
                raise ValueError("If `use_edges==True`, `edges_reducer` should not be None.")
            self._edges_aggregator = EdgesToGlobalsAggregator(edges_reducer)
        if self._use_nodes:
            if nodes_reducer is None:
                raise ValueError("If `use_nodes==True`, `nodes_reducer` should not be None.")
            self._nodes_aggregator = NodesToGlobalsAggregator(nodes_reducer)


        N_features = 0
        if model_info['out_dim'] == 'none':
            pre_features=graph.globals.shape[-1]
        else:
            pre_features= model_info['out_dim']
        if self._use_edges and graph.edges is not None:
            N_features += graph.edges.shape[-1]
        if self._use_nodes and graph.nodes is not None:
            N_features += graph.nodes.shape[-1]
        if self._use_globals and graph.globals is not None:
            N_features += graph.globals.shape[-1]
        if model_info['in_dim'] != 'none':
            N_features = model_info['in_dim']

        
        self._global_model = MLP(N_features, pre_features, model_info['dim'],
                                 model_info['n_blk'], model_info['norm'], model_info['activ'])
        
        if model_info['init_weight'] != 'none':
            utils.init_weights(self._global_model,model_info['init_weight'])         
        

    def forward(self, graph, global_model_kwargs=None):
        if global_model_kwargs is None:
            global_model_kwargs = {}
    
        globals_to_collect = []
    
        if self._use_edges and graph.edges is not None:
            _validate_graph(graph, (EDGES,), "when use_edges == True")
            globals_to_collect.append(self._edges_aggregator(graph).cuda())
        
        if self._use_nodes and graph.nodes is not None:
            _validate_graph(graph, (NODES,), "when use_nodes == True")
            globals_to_collect.append(self._nodes_aggregator(graph).cuda())
    
        if self._use_globals and graph.globals is not None:
            _validate_graph(graph, (GLOBALS,), "when use_globals == True")
            globals_to_collect.append(graph.globals.cuda())
    
        collected_globals = torch.cat(globals_to_collect, axis=-1)
        updated_globals = self._global_model(collected_globals, **global_model_kwargs)
        return graph.replace(globals=updated_globals)        
        
'''
Emodel_info ={ 'dim' : 32,
              'n_blk' : 4,
              'norm' : 'bn',
              'activ' : 'lrelu'
              }  
    
Eblock=EdgeBlock(Emodel_info, graphs_tuple)
Nblock=NodeBlock(Emodel_info, graphs_tuple)
Gblock=GlobalBlock(Emodel_info, graphs_tuple)

Eblock(graphs_tuple)
Nblock(graphs_tuple)
Gblock(graphs_tuple)
'''
