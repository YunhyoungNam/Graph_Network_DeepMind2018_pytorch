# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 18:13:07 2022

@author: darkstar0983
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from src import blocks
from src import utils_pt
import models
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn

def base_graph(n, d):
  """Define a basic mass-spring system graph structure.

  These are n masses (1kg) connected by springs in a chain-like structure. The
  first and last masses are fixed. The masses are vertically aligned at the
  start and are d meters apart; this is also the rest length for the springs
  connecting them. Springs have spring constant 50 N/m and gravity is 10 N in
  the negative y-direction.

  Args:
    n: number of masses
    d: distance between masses (as well as springs' rest length)

  Returns:
    data_dict: dictionary with globals, nodes, edges, receivers and senders
        to represent a structure like the one above.
  """
  # Nodes
  # Generate initial position and velocity for all masses.
  # The left-most mass has is at position (0, 0); other masses (ordered left to
  # right) have x-coordinate d meters apart from their left neighbor, and
  # y-coordinate 0. All masses have initial velocity 0m/s.
  nodes = np.zeros((n, 5), dtype=np.float32)
  half_width = d * n / 2.0
  nodes[:, 0] = np.linspace(
      -half_width, half_width, num=n, endpoint=False, dtype=np.float32)
  # indicate that the first and last masses are fixed
  nodes[(0, -1), -1] = 1.

  # Edges.
  edges, senders, receivers = [], [], []
  for i in range(n - 1):
    left_node = i
    right_node = i + 1
    # The 'if' statements prevent incoming edges to fixed ends of the string.
    if right_node < n - 1:
      # Left incoming edge.
      edges.append([50., d])
      senders.append(left_node)
      receivers.append(right_node)
    if left_node > 0:
      # Right incoming edge.
      edges.append([50., d])
      senders.append(right_node)
      receivers.append(left_node)

  return {
      "globals": [0., -10.],
      "nodes": nodes,
      "edges": edges,
      "receivers": receivers,
      "senders": senders
  }


def hookes_law(receiver_nodes, sender_nodes, k, x_rest):
  """Applies Hooke's law to springs connecting some nodes.

  Args:
    receiver_nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for the
      receiver node of each edge.
    sender_nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for the
      sender node of each edge.
    k: Spring constant for each edge.
    x_rest: Rest length of each edge.

  Returns:
    Nx2 Tensor of the force [f_x, f_y] acting on each edge.
  """
  diff = receiver_nodes[..., 0:2] - sender_nodes[..., 0:2]
  x = torch.linalg.vector_norm(diff, axis=-1, keepdims=True)
  force_magnitude = -1 * torch.multiply(k, (x - x_rest) / x)
  force = force_magnitude * diff
  return force

def euler_integration(nodes, force_per_node, step_size):
  """Applies one step of Euler integration.

  Args:
    nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for each node.
    force_per_node: Ex2 tf.Tensor of the force [f_x, f_y] acting on each edge.
    step_size: Scalar.

  Returns:
    A tf.Tensor of the same shape as `nodes` but with positions and velocities
        updated.
  """
  is_fixed = nodes[..., 4:5]
  # set forces to zero for fixed nodes
  force_per_node *= 1 - is_fixed
  new_vel = nodes[..., 2:4] + force_per_node * step_size
  return new_vel


class SpringMassSimulator(nn.Module):
    """Implements a basic Physics Simulator using the blocks library."""
    
    def __init__(self, step_size):
        super(SpringMassSimulator, self).__init__()
        self._step_size = step_size
      
        self._aggregator = blocks.ReceivedEdgesToNodesAggregator(reducer='sum')
    
    def forward(self, graph):
        """Builds a SpringMassSimulator.
      
        Args:
          graph: A graphs.GraphsTuple having, for some integers N, E, G:
              - edges: Nx2 tf.Tensor of [spring_constant, rest_length] for each
                edge.
              - nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for each
                node.
              - globals: Gx2 tf.Tensor containing the gravitational constant.
      
        Returns:
          A graphs.GraphsTuple of the same shape as `graph`, but where:
              - edges: Holds the force [f_x, f_y] acting on each edge.
              - nodes: Holds positions and velocities after applying one step of
                  Euler integration.
        """
        receiver_nodes = blocks.broadcast_receiver_nodes_to_edges(graph)
        sender_nodes = blocks.broadcast_sender_nodes_to_edges(graph)
      
        spring_force_per_edge = hookes_law(receiver_nodes, sender_nodes,
                                           graph.edges[..., 0:1],
                                           graph.edges[..., 1:2])
        graph = graph.replace(edges=spring_force_per_edge)
      
        spring_force_per_node = self._aggregator(graph)
        gravity = blocks.broadcast_globals_to_nodes(graph)
        updated_velocities = euler_integration(
            graph.nodes, spring_force_per_node + gravity, self._step_size)
        graph = graph.replace(nodes=updated_velocities)
        return graph
    

def prediction_to_next_state(input_graph, predicted_graph, step_size):
    # manually integrate velocities to compute new positions
    new_pos = input_graph.nodes[..., :2] + predicted_graph.nodes * step_size
    new_nodes = torch.cat(
        [new_pos, predicted_graph.nodes, input_graph.nodes[..., 4:5]], axis=-1)
    return input_graph.replace(nodes=new_nodes)


def roll_out_physics_gt(simulator, graph, steps, step_size):
    """Apply some number of steps of physical laws to an interaction network.
    
    Args:
      simulator: A SpringMassSimulator, or some module or callable with the same
        signature.
      graph: A graphs.GraphsTuple having, for some integers N, E, G:
          - edges: Nx2 tf.Tensor of [spring_constant, rest_length] for each edge.
          - nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for each
            node.
          - globals: Gx2 tf.Tensor containing the gravitational constant.
      steps: An integer.
      step_size: Scalar.
    
    Returns:
      A pair of:
      - The graph, updated after `steps` steps of simulation;
      - A `steps+1`xNx5 tf.Tensor of the node features at each step.
    """    
    
    def body(t, graph):
        predicted_graph = simulator(graph)
        if isinstance(predicted_graph, list):
          predicted_graph = predicted_graph[-1]
        graph = prediction_to_next_state(graph, predicted_graph, step_size)
        return t + 1, graph
    
    nodes_per_step = []
    nodes_per_step.append(graph.nodes)
    t = 1
    while t <= steps:
        
        
        _, graph=body(t, graph)
        nodes_per_step.append(graph.nodes)
        t += 1
    
    return graph, torch.stack(nodes_per_step)


def roll_out_physics_model(simulator, graph, num_processing_steps_tr, steps, step_size):
    """Apply some number of steps of physical laws to an interaction network.
    
    Args:
      simulator: A SpringMassSimulator, or some module or callable with the same
        signature.
      graph: A graphs.GraphsTuple having, for some integers N, E, G:
          - edges: Nx2 tf.Tensor of [spring_constant, rest_length] for each edge.
          - nodes: Ex5 tf.Tensor of [x, y, v_x, v_y, is_fixed] features for each
            node.
          - globals: Gx2 tf.Tensor containing the gravitational constant.
      steps: An integer.
      step_size: Scalar.
    
    Returns:
      A pair of:
      - The graph, updated after `steps` steps of simulation;
      - A `steps+1`xNx5 tf.Tensor of the node features at each step.
    """    
    
    def body(t, graph):
        predicted_graph = simulator(graph, num_processing_steps_tr)
        if isinstance(predicted_graph, list):
          predicted_graph = predicted_graph[-1]
        graph = prediction_to_next_state(graph, predicted_graph, step_size)
        return t + 1, graph
    
    nodes_per_step = []
    nodes_per_step.append(graph.nodes)
    t = 1
    while t <= steps:
        
        
        _, graph=body(t, graph)
        nodes_per_step.append(graph.nodes)
        t += 1
    
    return graph, torch.stack(nodes_per_step)



def apply_noise(graph, node_noise_level, edge_noise_level, global_noise_level):
  """Applies uniformly-distributed noise to a graph of a physical system.

  Noise is applied to:
  - the x and y coordinates (independently) of the nodes;
  - the spring constants of the edges;
  - the y coordinate of the global gravitational constant.

  Args:
    graph: a graphs.GraphsTuple having, for some integers N, E, G:
        - nodes: Nx5 Tensor of [x, y, _, _, _] for each node.
        - edges: Ex2 Tensor of [spring_constant, _] for each edge.
        - globals: Gx2 tf.Tensor containing the gravitational constant.
    node_noise_level: Maximum distance to perturb nodes' x and y coordinates.
    edge_noise_level: Maximum amount to perturb edge spring constants.
    global_noise_level: Maximum amount to perturb the Y component of gravity.

  Returns:
    The input graph, but with noise applied.
  """
  node_position_noise = torch.from_numpy(np.random.uniform(      
      low=-node_noise_level,
      high=node_noise_level,
      size=[graph.nodes.shape[0], 2])).type(graph.nodes.dtype)
  edge_spring_constant_noise = torch.from_numpy(np.random.uniform(      
      low=-edge_noise_level,
      high=edge_noise_level,
      size=[graph.edges.shape[0], 1])).type(graph.edges.dtype)
  global_gravity_y_noise = torch.from_numpy(np.random.uniform(
      low=-global_noise_level,
      high=global_noise_level,
      size = [graph.globals.shape[0], 1])).type(graph.globals.dtype)

  return graph.replace(
      nodes=torch.cat(
          [graph.nodes[..., :2] + node_position_noise, graph.nodes[..., 2:]],
          axis=-1),
      edges=torch.cat(
          [
              graph.edges[..., :1] + edge_spring_constant_noise,
              graph.edges[..., 1:]
          ],
          axis=-1),
      globals=torch.cat(
          [
              graph.globals[..., :1],
              graph.globals[..., 1:] + global_gravity_y_noise
          ],
          axis=-1))


def set_rest_lengths(graph):
    """Computes and sets rest lengths for the springs in a physical system.
    
    The rest length is taken to be the distance between each edge's nodes.
    
    Args:
      graph: a graphs.GraphsTuple having, for some integers N, E:
          - nodes: Nx5 Tensor of [x, y, _, _, _] for each node.
          - edges: Ex2 Tensor of [spring_constant, _] for each edge.
    
    Returns:
      The input graph, but with [spring_constant, rest_length] for each edge.
    """
    receiver_nodes = blocks.broadcast_receiver_nodes_to_edges(graph)
    sender_nodes = blocks.broadcast_sender_nodes_to_edges(graph)
    rest_length = torch.linalg.vector_norm(
        receiver_nodes[..., :2] - sender_nodes[..., :2], axis=-1, keepdims=True)
    return graph.replace(
        edges=torch.cat([graph.edges[..., :1], rest_length], axis=-1))


def generate_trajectory(simulator, graph, steps, step_size, node_noise_level,
                        edge_noise_level, global_noise_level):
  """Applies noise and then simulates a physical system for a number of steps.

  Args:
    simulator: A SpringMassSimulator, or some module or callable with the same
      signature.
    graph: a graphs.GraphsTuple having, for some integers N, E, G:
        - nodes: Nx5 Tensor of [x, y, v_x, v_y, is_fixed] for each node.
        - edges: Ex2 Tensor of [spring_constant, _] for each edge.
        - globals: Gx2 tf.Tensor containing the gravitational constant.
    steps: Integer; the length of trajectory to generate.
    step_size: Scalar.
    node_noise_level: Maximum distance to perturb nodes' x and y coordinates.
    edge_noise_level: Maximum amount to perturb edge spring constants.
    global_noise_level: Maximum amount to perturb the Y component of gravity.

  Returns:
    A pair of:
    - The input graph, but with rest lengths computed and noise applied.
    - A `steps+1`xNx5 tf.Tensor of the node features at each step.
  """
  graph = apply_noise(graph, node_noise_level, edge_noise_level,
                      global_noise_level)
  graph = set_rest_lengths(graph)
  _, n = roll_out_physics_gt(simulator, graph, steps, step_size)
  return graph, n