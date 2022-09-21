# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 19:20:12 2022

@author: darkstar0983
"""
##############################################
##
## GraphTuple of pytorch tensor implementation
##
##############################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from absl import logging
import graphs
import utils_np
import six
from six.moves import range
import torch
import tree
import einops


NODES = graphs.NODES
EDGES = graphs.EDGES
RECEIVERS = graphs.RECEIVERS
SENDERS = graphs.SENDERS
GLOBALS = graphs.GLOBALS
N_NODE = graphs.N_NODE
N_EDGE = graphs.N_EDGE

GRAPH_DATA_FIELDS = graphs.GRAPH_DATA_FIELDS
GRAPH_NUMBER_FIELDS = graphs.GRAPH_NUMBER_FIELDS
ALL_FIELDS = graphs.ALL_FIELDS

def index_select(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[N, L, D]
        indices: Tensor[N, K] or Tensor[N]
    Returns:
        Tensor[N, K, D] or Tensor[N, D]
    """
    if vectors.dim() < 3:
        out = vectors.index_select(index=indices, dim=0)
        return out
    else:
        N, L, D = vectors.shape
        squeeze = False
        if indices.ndim == 1:
            squeeze = True
            indices = indices.unsqueeze(-1)
        N2, K = indices.shape
        assert N == N2
        indices = einops.repeat(indices, "N K -> N K D", D=D)
        out = torch.gather(vectors, dim=1, index=indices.type(torch.int64))
        if squeeze:
            out = out.squeeze(1)
        return out

def unsorted_segment_cal(tensor, segment_ids, num_segments, mode = 'sum'):
    _, N_features = tensor.shape
    aggrated_list = []
    for i in range(num_segments):
        aggrated = tensor[segment_ids == i]
        if aggrated.shape[0] == 0:
            if tensor.device.type == 'cuda':
                aggrated = torch.zeros(1, N_features).cuda()
            else:
                aggrated = torch.zeros(1, N_features)
        if mode == 'sum':
            aggrated_list.append(torch.sum(aggrated, dim=0))   
        elif mode == 'mean':
            aggrated_list.append(torch.mean(aggrated, dim=0))   
        elif mode == 'max':
            aggrated_list.append(torch.max(aggrated, dim=0).values)   
        elif mode == 'min':
            aggrated_list.append(torch.min(aggrated, dim=0).values)   
    return torch.stack(aggrated_list,dim=0)  

def _to_compatible_data_dicts(data_dicts):
  """Convert the content of `data_dicts` to tensors of the right type.
  All fields are converted to `Tensor`s. The index fields (`SENDERS` and
  `RECEIVERS`) and number fields (`N_NODE`, `N_EDGE`) are cast to `tf.int32`.
  Args:
    data_dicts: An iterable of dictionaries with keys `ALL_KEYS` and
      values either `None`s, or quantities that can be converted to `Tensor`s.
  Returns:
    A list of dictionaries containing `Tensor`s or `None`s.
  """
  results = []
  for data_dict in data_dicts:
    result = {}
    for k, v in data_dict.items():
      if v is None:
        result[k] = None
      else:               
        #dtype = torch.int32 if k in [SENDERS, RECEIVERS, N_NODE, N_EDGE] else None
        dtype = torch.int32 if k in [SENDERS, RECEIVERS, N_NODE, N_EDGE] else torch.float32
        result[k] = torch.tensor(v, dtype=dtype)
        
    results.append(result)
  return results

def repeat(tensor, repeats, axis=0):
  """Repeats a `Tensor`'s elements along an axis by custom amounts.
  Equivalent to Numpy's `np.repeat`.
  `tensor and `repeats` must have the same numbers of elements along `axis`.
  Args:
    tensor: A `Tensor` to repeat.
    repeats: A 1D sequence of the number of repeats per element.
    axis: An axis to repeat along. Defaults to 0.
    name: (string, optional) A name for the operation.
    sum_repeats_hint: Integer with the total sum of repeats in case it is
      known at graph definition time.
  Returns:
    The `Tensor` with repeated values.
  """
  if tensor.device.type == 'cuda':
      repeats = repeats.cuda()
  if repeats.device.type == 'cuda':
      tensor = tensor.cuda()
  
  sum_repeats = torch.sum(repeats,dtype=torch.int32) #tf.reduce_sum(repeats)

  scatter_indices = (torch.cumsum(repeats, 0).roll(1,0)).type(torch.int32) #tf.cumsum(repeats, exclusive=True)
  scatter_indices[0] = 0   
  indices=scatter_indices.unsqueeze(1) #tf.expand_dims(scatter_indices, axis=1)
  updates=torch.ones_like(scatter_indices)  #tf.ones_like(scatter_indices)
  shape = sum_repeats + 1        
  
  if tensor.device.type == 'cuda':

      results = torch.zeros(shape, dtype=shape.dtype).cuda()      
  else:
      results = torch.zeros(shape, dtype=shape.dtype)
  
  results[indices.t().long()]=updates

  block_split_indicators = results[:-1]
#  block_split_indicators = tf.scatter_nd(
#        indices=tf.expand_dims(scatter_indices, axis=1),
#        updates=tf.ones_like(scatter_indices),
#        shape=[sum_repeats + 1])[:-1]
  gather_indices = torch.cumsum(block_split_indicators,0)-1#tf.cumsum(block_split_indicators, exclusive=False) - 1
  #gather_indices = gather_indices.type(torch.int32)
    # An alternative implementation of the same, where block split indicators
    # does not have an indicator for the first group, and requires less ops
    # but requires creating a matrix of size [len(repeats), sum_repeats] is:
    # cumsum_repeats = tf.cumsum(repeats, exclusive=False)
    # block_split_indicators = tf.reduce_sum(
    #     tf.one_hot(cumsum_repeats, sum_repeats, dtype=tf.int32), axis=0)
    # gather_indices = tf.cumsum(block_split_indicators, exclusive=False)

    # Now simply gather the tensor along the correct axis.
  #repeated_tensor = torch.gather(tensor, axis, gather_indices) #tf.gather(tensor, gather_indices, axis=axis)
  repeated_tensor = tensor.index_select(index=gather_indices.long(), dim=axis)
  #repeated_tensor = index_select(tensor, gather_indices.long())

  shape = list(tensor.shape)#tensor.shape.as_list()
  shape[axis] = sum_repeats #sum_repeats_hint
  repeated_tensor.view(shape)#repeated_tensor.set_shape(shape)  
  return repeated_tensor

def _compute_stacked_offsets(sizes, repeats):
  """Computes offsets to add to indices of stacked tensors (Tensorflow).
  When a set of tensors are stacked, the indices of those from the second on
  must be offset in order to be able to index into the stacked tensor. This
  computes those offsets.
  Args:
    sizes: A 1D `Tensor` of the sizes per graph.
    repeats: A 1D `Tensor` of the number of repeats per graph.
  Returns:
    A 1D `Tensor` containing the index offset per graph.
  """
  sizes = sizes[:-1].type(torch.int32)#tf.cast(tf.convert_to_tensor(sizes[:-1]), tf.int32)
  offset_values = (torch.cumsum(torch.cat([torch.zeros((1),dtype=torch.int32), sizes], 0),0)).type(torch.int32)#tf.cumsum(tf.concat([[0], sizes], 0))
  return repeat(offset_values, repeats)

def _populate_number_fields(data_dict):
  """Returns a dict with the number fields N_NODE, N_EDGE filled in.
  The N_NODE field is filled if the graph contains a non-`None` NODES field;
  otherwise, it is set to 0.
  The N_EDGE field is filled if the graph contains a non-`None` RECEIVERS field;
  otherwise, it is set to 0.
  Args:
    data_dict: An input `dict`.
  Returns:
    The data `dict` with number fields.
  """
  dct = data_dict.copy()
  for number_field, data_field in [[N_NODE, NODES], [N_EDGE, RECEIVERS]]:
    if dct.get(number_field) is None:
      if dct[data_field] is not None:
        dct[number_field] = torch.tensor(len(dct[data_field]),dtype=torch.int32) #tf.shape(dct[data_field])[0]
      else:
        dct[number_field] = torch.tensor(0,dtype=torch.int32)#tf.constant(0, dtype=tf.int32)
  return dct


def _concatenate_data_dicts(data_dicts):
  """Concatenate a list of data dicts to create the equivalent batched graph.
  Args:
    data_dicts: An iterable of data dictionaries with keys a subset of
      `GRAPH_DATA_FIELDS`, plus, potentially, a subset of `GRAPH_NUMBER_FIELDS`.
      Every element of `data_dicts` has to contain the same set of keys.
      Moreover, the key `NODES` or `N_NODE` must be present in every element of
      `data_dicts`.
  Returns:
    A data dictionary with the keys `GRAPH_DATA_FIELDS + GRAPH_NUMBER_FIELDS`,
    representing the concatenated graphs.
  Raises:
    ValueError: If two dictionaries in `data_dicts` have a different set of
      keys.
  """
  # Go from a list of dict to a dict of lists
  dct = collections.defaultdict(lambda: [])
  for data_dict in data_dicts:
    data_dict = _populate_number_fields(data_dict)
    for k, v in data_dict.items():
      if v is not None:
        dct[k].append(v)
      elif k not in dct:
        dct[k] = None
  dct = dict(dct)

  # Concatenate the graphs.
  for field, tensors in dct.items():
    if tensors is None:
      dct[field] = None
    elif field in list(GRAPH_NUMBER_FIELDS) + [GLOBALS]:
      dct[field] = torch.stack(tensors)
    else:
      dct[field] = torch.cat(tensors, axis=0)

  # Add offsets to the receiver and sender indices.
  if dct[RECEIVERS] is not None:
    offset = _compute_stacked_offsets(dct[N_NODE], dct[N_EDGE])
    dct[RECEIVERS] += offset
    dct[SENDERS] += offset

  return dct


def data_dicts_to_graphs_tuple(data_dicts:dict):
  """Creates a `graphs.GraphsTuple` containing tensors from data dicts.
   All dictionaries must have exactly the same set of keys with non-`None`
   values associated to them. Moreover, this set of this key must define a valid
   graph (i.e. if the `EDGES` are `None`, the `SENDERS` and `RECEIVERS` must be
   `None`, and `SENDERS` and `RECEIVERS` can only be `None` both at the same
   time). The values associated with a key must be convertible to `Tensor`s,
   for instance python lists, numpy arrays, or Tensorflow `Tensor`s.
   This method may perform a memory copy.
   The `RECEIVERS`, `SENDERS`, `N_NODE` and `N_EDGE` fields are cast to
   `np.int32` type.
  Args:
    data_dicts: An iterable of data dictionaries with keys in `ALL_FIELDS`.
    name: (string, optional) A name for the operation.
  Returns:
    A `graphs.GraphTuple` representing the graphs in `data_dicts`.
  """
  if isinstance(data_dicts, list):
      for data in data_dicts:
          assert isinstance(data, dict), 'Input is not dictionary'        
  else:
      assert isinstance(data_dicts, dict), 'Input is not dictionary'
  
  data_dicts = [dict(d) for d in data_dicts]
  for key in ALL_FIELDS:
    for data_dict in data_dicts:
      data_dict.setdefault(key, None)
  utils_np._check_valid_sets_of_keys(data_dicts)  # pylint: disable=protected-access
  
  data_dicts = _to_compatible_data_dicts(data_dicts)
  return graphs.GraphsTuple(**_concatenate_data_dicts(data_dicts))


'''
def get_num_graphs(input_graphs):
  """Returns the number of graphs (i.e. the batch size) in `input_graphs`.
  Args:
    input_graphs: A `graphs.GraphsTuple` containing tensors.
    name: (string, optional) A name for the operation.
  Returns:
    An `int` (if a static number of graphs is defined) or a `tf.Tensor` (if the
      number of graphs is dynamic).
  """
  return _get_shape(input_graphs.n_node)[0]


def _get_shape(tensor):
  """Returns the tensor's shape.
   Each shape element is either:
   - an `int`, when static shape values are available, or
   - a `tf.Tensor`, when the shape is dynamic.
  Args:
    tensor: A `tf.Tensor` to get the shape of.
  Returns:
    The `list` which contains the tensor's shape.
  """

  shape_list = list(tensor.shape)#tensor.shape.as_list()
  if all(s is not None for s in shape_list):
    return shape_list
  shape_tensor = torch.tensor(len(tensor),dtype=torch.int32) #tf.shape(tensor)
  return [shape_tensor[i] if s is None else s for i, s in enumerate(shape_list)]
'''

def get_num_graphs(input_graphs):
#    return _get_shape(input_graphs.n_node).item()
    return _get_shape(input_graphs.n_node)

def _get_shape(tensor):
    if tensor.device.type == 'cuda':
        return torch.tensor(len(tensor),dtype=torch.int32).cuda()
    else:
        return torch.tensor(len(tensor),dtype=torch.int32) #tf.shape(tensor
    #shape_list = list(tensor.shape)
    #if all(s is not None for s in shape_list):
    #  return shape_list    
 
def _validate_edge_fields_are_all_none(graph):
  if not all(getattr(graph, x) is None for x in [EDGES, RECEIVERS, SENDERS]):
    raise ValueError("Can only add fully connected a graph with `None`"
                     "edges, receivers and senders")    
 
def _create_complete_edges_from_nodes_dynamic(n_node, exclude_self_edges):
    """Creates complete edges for a graph with `n_node`.
    Args:
      n_node: (integer scalar `Tensor`) The number of nodes.
      exclude_self_edges: (bool) Excludes self-connected edges.
    Returns:
      A dict of RECEIVERS, SENDERS and N_EDGE data (`Tensor`s of rank 1).
    """
    rng = torch.arange(n_node)
    senders, receivers = torch.meshgrid(rng,rng)
    n_edge = n_node * n_node
    
    if exclude_self_edges:
        ind =(1 - torch.eye(n_node)).type(torch.bool)
        receivers=torch.masked_select(receivers, ind)
        senders = torch.masked_select(senders, ind)
        n_edge -= n_node
    
    receivers = torch.reshape(receivers.type(torch.int32), [n_edge])
    
    senders = torch.reshape(senders.type(torch.int32), [n_edge])
    n_edge = torch.reshape(n_edge, [1])
    
    return {RECEIVERS: receivers, SENDERS: senders, N_EDGE: n_edge} 

def fully_connect_graph_dynamic(graph,
                                exclude_self_edges=False):
    """Adds edges to a graph by fully-connecting the nodes.
    This method does not require the number of nodes per graph to be constant,
    or to be known at graph building time.
    Args:
      graph: A `graphs.GraphsTuple` with `None` values for the edges, senders and
        receivers.
      exclude_self_edges (default=False): Excludes self-connected edges.
      name: (string, optional) A name for the operation.
    Returns:
      A `graphs.GraphsTuple` containing `Tensor`s with fully-connected edges.
    Raises:
      ValueError: if any of the `EDGES`, `RECEIVERS` or `SENDERS` field is not
        `None` in `graph`.
    """

    _validate_edge_fields_are_all_none(graph)

    def body(i):
        edges = _create_complete_edges_from_nodes_dynamic(graph.n_node[i], exclude_self_edges)
        
        return i+1, edges
    
    num_graphs = get_num_graphs(graph)
    
    i = 0
    senders = []
    receivers = []
    n_edge = []
    while  i < num_graphs:
        
        i, edges = body(i)
        
        senders.append(edges['senders'])
        receivers.append(edges['receivers'])
        n_edge.append(edges['n_edge'])
        
    n_edge = torch.cat(n_edge)
    offsets = _compute_stacked_offsets(graph.n_node, n_edge)
    senders = torch.cat(senders) + offsets
    receivers = torch.cat(receivers) + offsets
    
    receivers=receivers.view(offsets.shape)
    senders=senders.view(offsets.shape)
    
    num_graphs = list(graph.n_node.shape)[0]
    n_edge=n_edge.view(num_graphs)
    
    return graph._replace(senders=senders, receivers=receivers, n_edge=n_edge)    
    


def _check_valid_index(index, element_name):
  """Verifies if a value with `element_name` is a valid index."""
  if isinstance(index, int):
    return True
  elif isinstance(index, torch.Tensor):
    if index.dtype != torch.int32 and index.dtype != torch.int64:
      raise TypeError(
          "Invalid tensor `{}` parameter. Valid tensor indices must have "
          "types tf.int32 or tf.int64, got {}."
          .format(element_name, index.dtype))
    if list(index.shape):
      raise TypeError(
          "Invalid tensor `{}` parameter. Valid tensor indices must be scalars "
          "with shape [], got{}"
          .format(element_name, list(index.shape)))
    return True
  else:
    raise TypeError(
        "Invalid `{}` parameter. Valid tensor indices must be integers "
        "or tensors, got {}."
        .format(element_name, type(index)))


def get_graph(input_graphs, index, name="get_graph"):
    """Indexes into a graph.
    Given a `graphs.graphsTuple` containing `Tensor`s and an index (either
    an `int` or a `slice`), index into the nodes, edges and globals to extract the
    graphs specified by the slice, and returns them into an another instance of a
    `graphs.graphsTuple` containing `Tensor`s.
    Args:
      input_graphs: A `graphs.GraphsTuple` containing `Tensor`s.
      index: An `int`, a `slice`, a tensor `int` or a tensor `slice`, to index
        into `graph`. `index` should be compatible with the number of graphs in
        `graphs`. The `step` parameter of the `slice` objects must be None.
      name: (string, optional) A name for the operation.
    Returns:
      A `graphs.GraphsTuple` containing `Tensor`s, made of the extracted
        graph(s).
    Raises:
      TypeError: if `index` is not an `int`, a `slice`, or corresponding tensor
        types.
      ValueError: if `index` is a slice and `index.step` if not None.
    """

    def safe_slice_none(value, slice_):
        if value is None:
          return value
        return value[slice_]
    
    if isinstance(index, (int, torch.Tensor)):
      _check_valid_index(index, "index")
      graph_slice = slice(index, index + 1)
    elif (isinstance(index, slice) and
          _check_valid_index(index.stop, "index.stop") and
          (index.start is None or _check_valid_index(
              index.start, "index.start"))):
      if index.step is not None:
        raise ValueError("slices with step/stride are not supported, got {}"
                         .format(index))
      graph_slice = index
    else:
      raise TypeError(
          "unsupported index type got {} with type {}. Index must be a valid "
          "scalar integer (tensor or int) or a slice of such values."
          .format(index, type(index)))
    
    start_slice = slice(0, graph_slice.start)
 
    start_node_index = torch.sum(
        input_graphs.n_node[start_slice])
    start_edge_index = torch.sum(
        input_graphs.n_edge[start_slice])
    end_node_index = start_node_index + torch.sum(
        input_graphs.n_node[graph_slice])
    end_edge_index = start_edge_index + torch.sum(
        input_graphs.n_edge[graph_slice])
    nodes_slice = slice(start_node_index, end_node_index)
    edges_slice = slice(start_edge_index, end_edge_index)
    
    sliced_graphs_dict = {}
    
    for field in set(GRAPH_NUMBER_FIELDS) | {"globals"}:
      sliced_graphs_dict[field] = safe_slice_none(
          getattr(input_graphs, field), graph_slice)
    
    field = "nodes"
    sliced_graphs_dict[field] = safe_slice_none(
        getattr(input_graphs, field), nodes_slice)
    
    for field in {"edges", "senders", "receivers"}:
      sliced_graphs_dict[field] = safe_slice_none(
          getattr(input_graphs, field), edges_slice)
      if (field in {"senders", "receivers"} and
          sliced_graphs_dict[field] is not None):
        sliced_graphs_dict[field] = sliced_graphs_dict[field] - start_node_index
    
    return graphs.GraphsTuple(**sliced_graphs_dict)


def _nested_concatenate(input_graphs, field_name, axis):
  """Concatenates a possibly nested feature field of a list of input graphs."""
  features_list = [getattr(gr, field_name) for gr in input_graphs
                   if getattr(gr, field_name) is not None]
  if not features_list:
    return None

  if len(features_list) < len(input_graphs):
    raise ValueError(
        "All graphs or no graphs must contain {} features.".format(field_name))

  #name = "concat_" + field_name
  return tree.map_structure(lambda *x: torch.cat(x, axis), *features_list)



def concat(input_graphs, axis):
    """Returns an op that concatenates graphs along a given axis.
    In all cases, the NODES, EDGES and GLOBALS dimension are concatenated
    along `axis` (if a fields is `None`, the concatenation is just a `None`).
    If `axis` == 0, then the graphs are concatenated along the (underlying) batch
    dimension, i.e. the RECEIVERS, SENDERS, N_NODE and N_EDGE fields of the tuples
    are also concatenated together.
    If `axis` != 0, then there is an underlying assumption that the receivers,
    SENDERS, N_NODE and N_EDGE fields of the graphs in `values` should all match,
    but this is not checked by this op.
    The graphs in `input_graphs` should have the same set of keys for which the
    corresponding fields is not `None`.
    Args:
      input_graphs: A list of `graphs.GraphsTuple` objects containing `Tensor`s
        and satisfying the constraints outlined above.
      axis: An axis to concatenate on.
      name: (string, optional) A name for the operation.
    Returns: An op that returns the concatenated graphs.
    Raises:
      ValueError: If `values` is an empty list, or if the fields which are `None`
        in `input_graphs` are not the same for all the graphs.
    """
    if not input_graphs:
      raise ValueError("List argument `input_graphs` is empty")
    utils_np._check_valid_sets_of_keys([gr._asdict() for gr in input_graphs])  # pylint: disable=protected-access
    if len(input_graphs) == 1:
      return input_graphs[0]
 
    nodes = _nested_concatenate(input_graphs, NODES, axis)
    edges = _nested_concatenate(input_graphs, EDGES, axis)
    globals_ = _nested_concatenate(input_graphs, GLOBALS, axis)

    output = input_graphs[0].replace(nodes=nodes, edges=edges, globals=globals_)
    if axis != 0:
      return output
    n_node_per_tuple = torch.stack(
        [torch.sum(gr.n_node) for gr in input_graphs])
    n_edge_per_tuple = torch.stack(
        [torch.sum(gr.n_edge) for gr in input_graphs])
    offsets = _compute_stacked_offsets(n_node_per_tuple, n_edge_per_tuple)
    n_node = torch.cat(
        [gr.n_node for gr in input_graphs], dim=0)
    n_edge = torch.cat(
        [gr.n_edge for gr in input_graphs], dim=0)
    receivers = [
        gr.receivers for gr in input_graphs if gr.receivers is not None
    ]
    receivers = receivers or None
    if receivers:
      receivers = torch.cat(receivers, axis) + offsets
    senders = [gr.senders for gr in input_graphs if gr.senders is not None]
    senders = senders or None
    if senders:
      senders = torch.cat(senders, axis) + offsets
    return output.replace(
        receivers=receivers, senders=senders, n_node=n_node, n_edge=n_edge)

def set_zero_node_features(graph,
                           node_size,
                           dtype=torch.float32):
    """Completes the node state of a graph.
    Args:
      graph: A `graphs.GraphsTuple` with a `None` edge state.
      node_size: (int) the dimension for the created node features.
      dtype: (tensorflow type) the type for the created nodes features.
      name: (string, optional) A name for the operation.
    Returns:
      The same graph but for the node field, which is a `Tensor` of shape
      `[number_of_nodes, node_size]`  where `number_of_nodes = sum(graph.n_node)`,
      with type `dtype`, filled with zeros.
    Raises:
      ValueError: If the `NODES` field is not None in `graph`.
      ValueError: If `node_size` is None.
    """
    if graph.nodes is not None:
        raise ValueError(
          "Cannot complete node state if the graph already has node features.")
    if node_size is None:
        raise ValueError("Cannot complete nodes with None node_size")
    
    n_nodes = torch.sum(graph.n_node) #tf.reduce_sum(graph.n_node)
    return graph._replace(
          nodes=torch.zeros((n_nodes, node_size), dtype=dtype))

def set_zero_edge_features(graph,
                           edge_size,
                           dtype=torch.float32):
    """Completes the edge state of a graph.
    Args:
      graph: A `graphs.GraphsTuple` with a `None` edge state.
      edge_size: (int) the dimension for the created edge features.
      dtype: (tensorflow type) the type for the created edge features.
      name: (string, optional) A name for the operation.
    Returns:
      The same graph but for the edge field, which is a `Tensor` of shape
      `[number_of_edges, edge_size]`, where `number_of_edges = sum(graph.n_edge)`,
      with type `dtype` and filled with zeros.
    Raises:
      ValueError: If the `EDGES` field is not None in `graph`.
      ValueError: If the `RECEIVERS` or `SENDERS` field are None in `graph`.
      ValueError: If `edge_size` is None.
    """
    if graph.edges is not None:
        raise ValueError(
          "Cannot complete edge state if the graph already has edge features.")
    if graph.receivers is None or graph.senders is None:
        raise ValueError(
          "Cannot complete edge state if the receivers or senders are None.")
    if edge_size is None:
        raise ValueError("Cannot complete edges with None edge_size")
    
    senders_leading_size = list(graph.senders.shape)[0]
    if senders_leading_size is not None:
        n_edges = senders_leading_size
    else:
        n_edges = torch.sum(graph.n_edge) #tf.reduce_sum(graph.n_edge)
    return graph._replace(
          edges=torch.zeros((n_edges, edge_size), dtype=dtype))


def set_zero_global_features(graph,
                             global_size,
                             dtype=torch.float32):
    """Completes the global state of a graph.
    Args:
      graph: A `graphs.GraphsTuple` with a `None` global state.
      global_size: (int) the dimension for the created global features.
      dtype: (tensorflow type) the type for the created global features.
      name: (string, optional) A name for the operation.
    Returns:
      The same graph but for the global field, which is a `Tensor` of shape
      `[num_graphs, global_size]`, type `dtype` and filled with zeros.
    Raises:
      ValueError: If the `GLOBALS` field of `graph` is not `None`.
      ValueError: If `global_size` is not `None`.
    """
    if graph.globals is not None:
        raise ValueError(
          "Cannot complete global state if graph already has global features.")
    if global_size is None:
        raise ValueError("Cannot complete globals with None global_size")
    
    n_graphs = get_num_graphs(graph)
    return graph._replace(
          globals=torch.zeros((n_graphs, global_size), dtype=dtype))



def graphs_tuple_pt_to_graphs_tuple(graphs_tuple_pt):
    graphs_tuple=graphs_tuple_pt.replace(nodes=graphs_tuple_pt.nodes.numpy(),
                            edges=graphs_tuple_pt.edges.numpy(),
                            globals=graphs_tuple_pt.globals.numpy(),
                            receivers=graphs_tuple_pt.receivers.numpy(),
                            senders=graphs_tuple_pt.senders.numpy(),
                            n_node=graphs_tuple_pt.n_node.numpy(),
                            n_edge=graphs_tuple_pt.n_edge.numpy())    
    return graphs_tuple


def graphs_tuple_to_graphs_tuple_pt(graphs_tuple):
    graphs_tuple_pt=graphs_tuple.replace(nodes=torch.from_numpy(graphs_tuple.nodes).float(),
                            edges=torch.from_numpy(graphs_tuple.edges).float(),
                            globals=torch.from_numpy(graphs_tuple.globals).float(),
                            receivers=torch.from_numpy(graphs_tuple.receivers),
                            senders=torch.from_numpy(graphs_tuple.senders),
                            n_node=torch.from_numpy(graphs_tuple.n_node),
                            n_edge=torch.from_numpy(graphs_tuple.n_edge))    
    return graphs_tuple_pt


def graphs_to_cuda(graph):
    graph = graph.replace(edges = graph.edges.cuda() if graph.edges is not None else graph.edges,
                          nodes = graph.nodes.cuda() if graph.nodes is not None else graph.nodes,
                          globals = graph.globals.cuda() if graph.globals is not None else graph.globals,
                          senders = graph.senders.cuda() if graph.senders is not None else graph.senders,
                          receivers = graph.receivers.cuda() if graph.receivers is not None else graph.receivers,
                          n_node = graph.n_node.cuda() if graph.n_node is not None else graph.n_node,
                          n_edge = graph.n_edge.cuda() if graph.n_edge is not None else graph.n_edge) 
    return graph


def graphs_to_cpu(graph):
    graph = graph.replace(edges = graph.edges.detach().cpu() if graph.edges is not None else graph.edges,
                          nodes = graph.nodes.detach().cpu() if graph.nodes is not None else graph.nodes,
                          globals = graph.globals.detach().cpu() if graph.globals is not None else graph.globals,
                          senders = graph.senders.detach().cpu() if graph.senders is not None else graph.senders,
                          receivers = graph.receivers.detach().cpu() if graph.receivers is not None else graph.receivers,
                          n_node = graph.n_node.detach().cpu() if graph.n_node is not None else graph.n_node,
                          n_edge = graph.n_edge.detach().cpu() if graph.n_edge is not None else graph.n_edge) 
    return graph
    