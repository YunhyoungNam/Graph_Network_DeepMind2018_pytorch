# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:55:33 2022

@author: darkstar0983
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src import modules
from src import utils_pt

import torch.nn as nn
'''
model infomation dict
'''


class MLPGraphIndependent(nn.Module):
    """GraphIndependent with MLP edge, node, and global models."""    

    def __init__(self, graph, edge_model_info, node_model_info, global_model_info):
        super(MLPGraphIndependent, self).__init__()
        
        self._network = modules.GraphIndependent(graph,
            edge_model_info,
            node_model_info,
            global_model_info)
    
    def forward(self, inputs):
        return self._network(inputs)
    

class MLPGraphNetwork(nn.Module):
    """GraphNetwork with MLP edge, node, and global models."""

    def __init__(self, graph, edge_model_info, node_model_info, global_model_info):
        super(MLPGraphNetwork, self).__init__()
        
        self._network = modules.GraphNetwork(edge_model_info,
                                             node_model_info,
                                             global_model_info,
                                             graph)

    def forward(self, inputs):
        return self._network(inputs)
    
    
class EncodeProcessDecode(nn.Module):
    """Full encode-process-decode model.
    The model we explore includes three components:
    - An "Encoder" graph net, which independently encodes the edge, node, and
      global attributes (does not compute relations etc.).
    - A "Core" graph net, which performs N rounds of processing (message-passing)
      steps. The input to the Core is the concatenation of the Encoder's output
      and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
      the processing step).
    - A "Decoder" graph net, which independently decodes the edge, node, and
      global attributes (does not compute relations etc.), on each message-passing
      step.
                        Hidden(t)   Hidden(t+1)
                           |            ^
              *---------*  |  *------*  |  *---------*
              |         |  |  |      |  |  |         |
    Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
              |         |---->|      |     |         |
              *---------*     *------*     *---------*
    """

    def __init__(self, opts):
        super(EncodeProcessDecode, self).__init__()
        
        self.opts = opts
        
        self._encoder = MLPGraphIndependent(opts.en_graph,
                                            opts.en_edge_model_info,
                                            opts.en_node_model_info,
                                            opts.en_global_model_info)
        self._core = MLPGraphNetwork(opts.co_graph,
                                     opts.co_edge_model_info,
                                     opts.co_node_model_info,
                                     opts.co_global_model_info)
        self._decoder = MLPGraphIndependent(opts.de_graph,
                                            opts.de_edge_model_info,
                                            opts.de_node_model_info,
                                            opts.de_global_model_info)
        # Transforms the outputs into the appropriate shapes.
        self._output_transform = modules.GraphIndependent(opts.tr_graph,
                                                          opts.tr_edge_model_info,
                                                          opts.tr_node_model_info,
                                                          opts.tr_global_model_info)

            
    
    def forward(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)
        latent0 = latent
        output_ops = []
        for _ in range(num_processing_steps):
          core_input = utils_pt.concat([latent0, latent], axis=1)
          latent = self._core(core_input)
          decoded_op = self._decoder(latent)
          output_ops.append(self._output_transform(decoded_op))
        return output_ops    