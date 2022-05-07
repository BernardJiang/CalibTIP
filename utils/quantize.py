from collections import namedtuple
import torch
import torch.nn as nn
from models.modules.quantize import calculate_qparams, quantize, QConv2d,QLinear

QTensor = namedtuple('QTensor', ['tensor', 'stepsize', 'zero_point'])


def get_broadcastshape(x):
    shp = x.shape
    p1 = torch.ones(len(shp), dtype=torch.int, device=x.device)
    p1[0] = x.shape[0]
    p2 = p1.tolist()
    p3 = tuple(p2)
    return p3
    
def quantize_tensor(x, scale, qmin, qmax, two_power_of_radix):
    # newshape = get_broadcastshape(x)
    # abs_max_val = torch.max(torch.abs(torch.reshape(x, (x.shape[0], -1))), 1)[0]
    # zero_point = 0
    # stepsize = torch.reshape(abs_max_val / qmax, newshape)
    
    # q_x = x / stepsize
    # q_x.clamp_(qmin, qmax).round_()
    # q_x = q_x.to(torch.int8)
    # return QTensor(tensor=q_x, stepsize=stepsize, zero_point=zero_point)
    
    # radix = torch.floor( torch.log2( (2.**(num_bits-1))/abs_max_val )).to(torch.int8)
    # radix = torch.reshape(radix, newshape)
    # q_x = x * (2.**radix) 
    # q_x.clamp_(qmin, qmax).round_()
    # q_x = q_x.to(torch.int8)
    # return QTensor(tensor=q_x, stepsize=radix, zero_point=zero_point)  #re-use the stepsize as redix
    x.mul_(scale)
    x.mul_(two_power_of_radix)
    x.clamp_(qmin, qmax).round_()
    return x


def dequantize_tensor(q_x, scale, qmin, qmax, two_power_of_radix):
    # return q_x.stepsize * q_x.tensor.float() 
    # return q_x.tensor.float() / (2.**q_x.stepsize )  #re-use stepsize as radix.
    return q_x.div_(two_power_of_radix).div_(scale)
    

def quantize_model(model):
    qparams = {}

    for n, p in model.state_dict().items():
        qp = quantize_tensor(p)
        qparams[n + '.quantization.stepsize'] = torch.FloatTensor([qp.stepsize])
        qparams[n + '.quantization.zero_point'] = torch.ByteTensor([qp.zero_point])
        p.copy_(qp.tensor)
    model.type('torch.ByteTensor')
    for n, p in qparams.items():
        model.register_buffer(n, p)
    model.quantized = True


def dequantize_model(model):
    model.float()
    params = model.state_dict()
    for n, p in params.items():
        if 'quantization' not in n:
            qp = QTensor(tensor=p,
                         stepsize=params[n + '.quantization.stepsize'][0],
                         zero_point=params[n + '.quantization.zero_point'][0])
            p.copy_(dequantize_tensor(qp))
            model.register_buffer(n + '.quantization.stepsize', None)
            model.register_buffer(n + '.quantization.zero_point', None)
            
    model.quantized = None

def is_q_module(m):
    return isinstance(m, QConv2d) or isinstance(m, QLinear)



def tensor_fl2fx2fl(x, num_bits=8):
    return dequantize_tensor(quantize_tensor(x, num_bits))


def get_quantized_model_and_params(model, dequantflag=True, qparams = {}):    
    for i,m in enumerate(model.children()):
        if is_q_module(m):
            with torch.no_grad():
                # dqw = tensor_fl2fx2fl(m.weight, num_bits=m.quantize_weight.num_bits)
                inshape = (-1, 1, 1)
                weightoutshape = (-1, 1, 1, 1)
                weightinshape = (1, -1, 1, 1)
                if isinstance(m, nn.Conv2d):
                    in_channels = m.in_channels 
                    out_channels = m.out_channels 
                    if m.groups != 1: 
                        weightinshape = (-1, 1, 1, 1)
                else: # isinstance(m, nn.Linear):
                    in_channels = m.in_features
                    out_channels = m.out_features
                    inshape = (-1)
                    weightoutshape = (-1, 1)
                    weightinshape = (1, -1)
                weight_scale = m.quantize_weight.running_scale.reshape(weightoutshape) / m.quantize_input.running_scale.reshape(weightinshape)
                qw = quantize_tensor(m.weight, weight_scale, m.quantize_weight.qmin, m.quantize_weight.qmax, m.quantize_weight.two_power_of_radix)
                if dequantflag:
                    dqw = dequantize_tensor(m.weight, weight_scale, m.quantize_weight.qmin, m.quantize_weight.qmax, m.quantize_weight.two_power_of_radix)
                else:
                    dqw = qw
                m.weight.copy_(dqw)
                
                scales = weight_scale.flatten().tolist()
                num_bits = (torch.log2(m.quantize_weight.qmax + 1) + 1).flatten().tolist()
                radixes = torch.log2(m.quantize_weight.two_power_of_radix).flatten().tolist()
                qparams[m.name+'.weight'] = {                        
                        'scale':  scales,
                        'radix':  radixes,
                        'bitwidth': num_bits,
                    }
               
                if m.bias is not None:
                    qb = quantize_tensor(m.bias, m.quantize_weight.running_scale, m.quantize_weight.bias_qmin, m.quantize_weight.bias_qmax, m.quantize_weight.bias_two_power_of_radix)
                    if dequantflag:
                        dqb = dequantize_tensor(m.bias, m.quantize_weight.running_scale, m.quantize_weight.bias_qmin, m.quantize_weight.bias_qmax, m.quantize_weight.bias_two_power_of_radix)
                    else:
                        dqb = qb
                    m.bias.copy_(dqb)
                    
                    bias_scales = m.quantize_weight.running_scale.flatten().tolist()
                    bias_num_bits = (torch.log2(m.quantize_weight.bias_qmax + 1) + 1).flatten().tolist()
                    bias_radixes = torch.log2(m.quantize_weight.bias_two_power_of_radix).flatten().tolist()
                    qparams[m.name+'.bias'] = {                        
                        'scale':  bias_scales,
                        'radix':  bias_radixes,
                        'bitwidth': bias_num_bits,
                    }
                
        qparams = get_quantized_model_and_params(m, dequantflag, qparams)
        
    qparams["input"] =  {
        "scale": {
            "all": 1.0
        },
        "radix": {
            "all": 5.0
        },
        "bitwidth": {
        "all": 8.0
        }
    }

    model.quantized = None
    return qparams

def get_quantized_params(model, qparams = {}):    
    for i,m in enumerate(model.children()):
        if is_q_module(m):
            with torch.no_grad():

                in_scales = m.quantize_input.running_scale.flatten().tolist()
                in_num_bits = (torch.log2(m.quantize_input.qmax + 1) + 1).flatten().tolist()
                in_radixes = torch.log2(m.quantize_input.two_power_of_radix).flatten().tolist()
                qparams[m.name+'.input'] = {                        
                    'scale':  in_scales,
                    'radix':  in_radixes,
                    'bitwidth': in_num_bits,
                }

                num_bits = (torch.log2(m.quantize_weight.qmax + 1) + 1).flatten().tolist()
                radixes = torch.log2(m.quantize_weight.two_power_of_radix).flatten().tolist()
                qparams[m.name+'.weight'] = {                        
                        # 'scale':  scales,
                        'radix':  radixes,
                        'bitwidth': num_bits,
                    }
               
                if m.bias is not None:
                    
                    bias_scales = m.quantize_weight.running_scale.flatten().tolist()
                    bias_num_bits = (torch.log2(m.quantize_weight.bias_qmax + 1) + 1).flatten().tolist()
                    bias_radixes = torch.log2(m.quantize_weight.bias_two_power_of_radix).flatten().tolist()
                    qparams[m.name+'.output'] = {                        
                        'scale':  bias_scales,
                        'radix':  bias_radixes,
                        'bitwidth': bias_num_bits,
                    }
                
        qparams = get_quantized_params(m, qparams)
        
    qparams["input"] =  {
        "scale": {
            "all": 1.0
        },
        "radix": {
            "all": 5.0
        },
        "bitwidth": {
        "all": 8.0
        }
    }

    model.quantized = None
    return qparams
