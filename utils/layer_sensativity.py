import torch
from torch.utils import data
import torch.nn as nn
from models.modules.quantize import calculate_qparams, quantize, QConv2d,QLinear
import numpy as np
import torch.fx as fx
from torch.fx.node import Argument, Target
from torch.nn.utils.fusion import fuse_conv_bn_eval
from typing import Type, Dict, Any, Tuple, Iterable, Optional, List, cast
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp
import copy
from collections import defaultdict
import torch.utils.mkldnn as th_mkldnn
import operator
import time
import logging
from enum import Enum



def search_replace_layer(model,all_names,num_bits_activation,num_bits_weight,name_model=''):
    for i,m in enumerate(model.children()):
        modules_names=[key for key in model._modules.keys()]
        layer_name=name_model+'.'+modules_names[i] if name_model !='' else name_model+modules_names[i]
        m.name=layer_name
        if layer_name in all_names:
            print("Layer {}, precision switch from w{}a{} to w{}a{}.".format(
                layer_name, m.num_bits_weight, m.num_bits, num_bits_weight, num_bits_activation))

            m.num_bits=num_bits_activation
            m.num_bits_weight = num_bits_weight
            m.quantize_input.num_bits=num_bits_activation
            m.quantize_weight.num_bits=num_bits_weight
 
        search_replace_layer(m,all_names,num_bits_activation,num_bits_weight,layer_name)
    return model

def search_replace_layer_name(model,all_names,name_model=''):
    for i,m in enumerate(model.children()):
        modules_names=[key for key in model._modules.keys()]
        layer_name=name_model+'.'+modules_names[i] if name_model !='' else name_model+modules_names[i]
        m.name=layer_name
        # for display correct bitwidth.
        if layer_name in all_names:
            m.quantize_input.num_bits = int(torch.log2(m.quantize_input.qmax+1).item()+1)
            m.quantize_weight.num_bits = int(torch.log2(m.quantize_weight.qmax+1).item()+1)            
            m.num_bits = m.quantize_input.num_bits
            m.num_bits_weight = m.quantize_weight.num_bits

        # print("Assigned layer name {}.".format(layer_name))
        search_replace_layer_name(m, all_names, layer_name)
    return model

# "{'conv1': [8, 8], 'layer1.0.conv1': [8, 8], 'layer1.0.conv2': [4, 4], 'layer1.0.conv3': [4, 4], 'layer1.0.downsample.0': [8, 8], 'layer1.1.conv1': [4, 4], 'layer1.1.conv2': [4, 4], 'layer1.1.conv3': [4, 4], 'layer1.2.conv1': [4, 4], 'layer1.2.conv2': [4, 4], 'layer1.2.conv3': [4, 4], 'layer2.0.conv1': [4, 4], 'layer2.0.conv2': [2, 2], 'layer2.0.conv3': [4, 4], 'layer2.0.downsample.0': [4, 4], 'layer2.1.conv1': [2, 2], 'layer2.1.conv2': [4, 4], 'layer2.1.conv3': [4, 4], 'layer2.2.conv1': [2, 2], 'layer2.2.conv2': [2, 2], 'layer2.2.conv3': [4, 4], 'layer2.3.conv1': [2, 2], 'layer2.3.conv2': [2, 2], 'layer2.3.conv3': [4, 4], 'layer3.0.conv1': [4, 4], 'layer3.0.conv2': [2, 2], 'layer3.0.conv3': [2, 2], 'layer3.0.downsample.0': [2, 2], 'layer3.1.conv1': [2, 2], 'layer3.1.conv2': [2, 2], 'layer3.1.conv3': [2, 2], 'layer3.2.conv1': [2, 2], 'layer3.2.conv2': [2, 2], 'layer3.2.conv3': [2, 2], 'layer3.3.conv1': [2, 2], 'layer3.3.conv2': [2, 2], 'layer3.3.conv3': [2, 2], 'layer3.4.conv1': [2, 2], 'layer3.4.conv2': [2, 2], 'layer3.4.conv3': [2, 2], 'layer3.5.conv1': [2, 2], 'layer3.5.conv2': [2, 2], 'layer3.5.conv3': [2, 2], 'layer4.0.conv1': [2, 2], 'layer4.0.conv2': [2, 2], 'layer4.0.conv3': [2, 2], 'layer4.0.downsample.0': [2, 2], 'layer4.1.conv1': [2, 2], 'layer4.1.conv2': [2, 2], 'layer4.1.conv3': [2, 2], 'layer4.2.conv1': [2, 2], 'layer4.2.conv2': [2, 2], 'layer4.2.conv3': [2, 2], 'fc': [4, 4]}"
def search_replace_layer_from_dict(model, layers_precision_dict, name_model=''):
    for i,m in enumerate(model.children()):
        modules_names=[key for key in model._modules.keys()]
        layer_name=name_model+'.'+modules_names[i] if name_model !='' else name_model+modules_names[i]
        m.name=layer_name
        if layer_name in layers_precision_dict:
            new_prec = layers_precision_dict[layer_name]
            print("Layer {}, precision switch from w{}a{} to w{}a{}.".format(
                layer_name, m.num_bits_weight, m.num_bits, new_prec[0], new_prec[1]))
            m.num_bits=new_prec[1]
            m.num_bits_weight = new_prec[0]
            m.quantize_input.num_bits=new_prec[1]
            m.quantize_weight.num_bits=new_prec[0]
        search_replace_layer_from_dict(m,layers_precision_dict,layer_name)
    return model

def search_replace_layer_from_json(model, onnx_model, layers_precision_json, name_model=''):
    for i,m in enumerate(model.children()):
        modules_names=[key for key in model._modules.keys()]
        layer_name=name_model+'.'+modules_names[i] if name_model !='' else name_model+modules_names[i]
        m.name=layer_name
        if layer_name in layers_precision_json:
            new_prec = layers_precision_json[layer_name]

            wbits = new_prec["weight_bitwidth"]
            dbits = new_prec["input_datapath_bitwidth"]
            bbits = new_prec["bias_bitwidth"]

            print("Json : Layer {}, precision switch from w{}a{} to w{}a{}b{}.".format(
                layer_name, m.num_bits_weight, m.num_bits, wbits, dbits, bbits))
                        
            m.num_bits=dbits
            m.num_bits_weight = wbits
            m.quantize_input.num_bits = dbits
            m.quantize_weight.num_bits = wbits
            
            #new implemention:
            dev = next(m.parameters()).device
            inshape = (-1, 1, 1)
            weightoutshape = (-1, 1, 1, 1)
            weightinshape = (1, -1, 1, 1)
            if isinstance(m, nn.Conv2d):
                if m.groups != 1: 
                    weightinshape = (-1, 1, 1, 1)
                
            if isinstance(m, nn.Linear):
                inshape = (-1)
                weightoutshape = (-1, 1)
                weightinshape = (1, -1)
                
            data_scale = torch.tensor(new_prec["input_scale"]).reshape(inshape).to(dev)
            data_qmin = torch.tensor(-(2.**(dbits-1) - 1.)).to(dev)
            data_qmax = torch.tensor(2.**(dbits-1) - 1.).to(dev)
            data_two_power_of_radix = torch.tensor(2.** np.array(new_prec["input_datapath_radix"])).reshape(inshape).to(dev)
            
            # scale_out = np.array(new_prec["output_scale"]).reshape(weightoutshape)
            # scale_in  = np.array(new_prec["input_scale"][0]).reshape(weightinshape)
            # weight_scale = torch.tensor(scale_out/scale_in ).to(dev)
            weight_qmin = torch.tensor(-(2.**(wbits-1) - 1.)).to(dev)
            weight_qmax = torch.tensor(2.**(wbits-1) - 1.).to(dev)
            weight_two_power_of_radix = torch.tensor(2.** np.array(new_prec["weight_radix"])).reshape(weightoutshape).to(dev)

            bias_scale = torch.tensor(new_prec["output_scale"]).to(dev)
            bias_qmin = torch.tensor(-(2.**(bbits-1) - 1.)).to(dev)
            bias_qmax = torch.tensor(2.**(bbits-1) - 1.).to(dev)
            bias_two_power_of_radix = torch.tensor(2.** np.array(new_prec["bias_radix"])).to(dev)
            
            m.quantize_input.register_parameter('running_scale', nn.Parameter(data_scale))
            m.quantize_input.register_parameter('qmin',  nn.Parameter(data_qmin))
            m.quantize_input.register_parameter('qmax',  nn.Parameter(data_qmax))
            m.quantize_input.register_parameter('two_power_of_radix',  nn.Parameter(data_two_power_of_radix))

            # m.quantize_weight.register_parameter('scale', nn.Parameter(weight_scale))
            m.quantize_weight.register_parameter('qmin',  nn.Parameter(weight_qmin))
            m.quantize_weight.register_parameter('qmax',  nn.Parameter(weight_qmax))
            m.quantize_weight.register_parameter('two_power_of_radix',  nn.Parameter(weight_two_power_of_radix))

            m.quantize_weight.register_parameter('running_scale', nn.Parameter(bias_scale))
            m.quantize_weight.register_parameter('bias_qmin',  nn.Parameter(bias_qmin))
            m.quantize_weight.register_parameter('bias_qmax',  nn.Parameter(bias_qmax))
            m.quantize_weight.register_parameter('bias_two_power_of_radix',  nn.Parameter(bias_two_power_of_radix))
           
        search_replace_layer_from_json(m,onnx_model, layers_precision_json, layer_name)
    return model

def is_q_module(m):
    return isinstance(m, QConv2d) or isinstance(m, QLinear)

def extract_all_quant_layers_names(model,q_names=[],name_model=''):
    for i,m in enumerate(model.children()):
        modules_names=[key for key in model._modules.keys()]
        layer_name=name_model+'.'+modules_names[i] if name_model !='' else name_model+modules_names[i]
        m.name=layer_name
        if is_q_module(m):
            q_names.append(m.name)
            print("Layer {}, if in FP32.".format(layer_name))
        q_names = check_quantized_model(m,q_names,layer_name)
    return q_names  
    
def check_quantized_model(model,fp_names=[],name_model=''):
    for i,m in enumerate(model.children()):
        modules_names=[key for key in model._modules.keys()]
        layer_name=name_model+'.'+modules_names[i] if name_model !='' else name_model+modules_names[i]
        m.name=layer_name
        if (is_q_module(m) and m.measure) or not is_q_module:
            fp_names.append(m.name)
            print("Layer {}, if in FP32.".format(layer_name))
        fp_names = check_quantized_model(m,fp_names,layer_name)
    return fp_names  

def extract_save_quant_state_dict(model,all_names,filename='int_state_dict.pth.tar'):
    
    state_dict=model.state_dict()
    
    for key in state_dict.keys():
        #import pdb; pdb.set_trace()
        val=state_dict[key]
        if 'weight' in key:
            num_bits = 4 if key[:-7] in all_names else 8
            if num_bits==4:
                import pdb; pdb.set_trace()
            weight_qparams = calculate_qparams(val, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=None)
            val_q=quantize(val, qparams=weight_qparams,dequantize=False) 
            zero_point=(-weight_qparams[1]/weight_qparams[0]*(2**weight_qparams[2]-1)).round()
            val_q=val_q-zero_point
            print(val_q.eq(0).sum().float().div(val_q.numel()))
        if 'bias' in key:
            val_q = quantize(val, num_bits=num_bits*2,flatten_dims=(0, -1))  

        state_dict[key] = val_q
    torch.save(state_dict,filename)        
    return state_dict  


def _parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

# Works for length 2 patterns with 2 modules
def matches_module_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]):
    if len(node.args) == 0:
        return False
    nodes: Tuple[Any, fx.Node] = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)

def fuse(model: torch.nn.Module, inplace=False) -> torch.nn.Module:
    """
    Fuses convolution/BN layers for inference purposes. Will deepcopy your
    model by default, but can modify the model inplace as well.
    """
    patterns = [(nn.Conv2d, nn.Conv2d),
                (nn.Conv2d, nn.BatchNorm3d)]
    if not inplace:
        model = copy.deepcopy(model)
    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                # fused_conv = fuse_conv_bn_eval(conv, bn)
                # replace_node_module(node.args[0], modules, fused_conv)
                # node.replace_all_uses_with(node.args[0])
                # new_graph.erase_node(node)
    return fx.GraphModule(fx_model, new_graph)

def get_seq_exec_list(model):
    DUMMY_INPUT = torch.randn(1,3,224,224)
    model.eval()
    traced = torch.jit.trace(model, (DUMMY_INPUT,), check_trace=False)
    seq_exec_list = traced.code
    seq_exec_list = seq_exec_list.split('\n')
    for idx, item in enumerate(seq_exec_list):
        print("[{}]: {}".format(idx, item))
        
    gm = torch.fx.symbolic_trace(model)
    for n in gm.graph.nodes:
        flag = False
        if n.op == 'call_module':
            flag = isinstance(n.target, nn.Conv2d) # or isinstance(n, nn.Linear)
        
        print(f'{n.name} = {n.op} target={n.target} args={n.args} isConvorLinear = {flag}')

    # fuse(model)
    return
