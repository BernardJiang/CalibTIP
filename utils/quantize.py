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
    
def quantize_tensor(x, num_bits=8):
    qmin = -(2.**(num_bits-1) - 1.)
    qmax = 2.**(num_bits-1) - 1.
    newshape = get_broadcastshape(x)
    abs_max_val = torch.max(torch.abs(torch.reshape(x, (x.shape[0], -1))), 1)[0]
    zero_point = 0
    stepsize = torch.reshape(abs_max_val / qmax, newshape)
    
    # q_x = x / stepsize
    # q_x.clamp_(qmin, qmax).round_()
    # q_x = q_x.to(torch.int8)
    # return QTensor(tensor=q_x, stepsize=stepsize, zero_point=zero_point)
    
    radix = torch.floor( torch.log2( (2.**(num_bits-1))/abs_max_val )).to(torch.int8)
    radix = torch.reshape(radix, newshape)
    q_x = x * (2.**radix) 
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.to(torch.int8)
    return QTensor(tensor=q_x, stepsize=radix, zero_point=zero_point)  #re-use the stepsize as redix

def dequantize_tensor(q_x):
    # return q_x.stepsize * q_x.tensor.float() 
    return q_x.tensor.float() / (2.**q_x.stepsize )  #re-use stepsize as radix.

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


def get_quantized_model_and_params(model, qparams = {}):    
    for i,m in enumerate(model.children()):
        if is_q_module(m):
            with torch.no_grad():
                # dqw = tensor_fl2fx2fl(m.weight, num_bits=m.quantize_weight.num_bits)
                qw = quantize_tensor(m.weight, num_bits=m.quantize_weight.num_bits)
                dqw = dequantize_tensor(qw)
                
                m.weight.copy_(dqw)
                radix = qw.stepsize #re-use stepsize as radix
                
                qmax = 2.**(m.quantize_input.num_bits-1)
                radix_input = torch.floor(torch.log2(qmax/m.quantize_input.running_range)).to(torch.int8)
                
                
                qparams[m.name] = {
                        'shape': list(m.weight.shape),
                        
                        'num_bits': m.quantize_weight.num_bits,
                        'range': m.quantize_weight.running_range.flatten().tolist(),
                        'zero_point': m.quantize_weight.running_zero_point.flatten().tolist(),
                        
                        'num_bits_input': m.quantize_input.num_bits,
                        'range_input': m.quantize_input.running_range.flatten().tolist(),
                        'zero_point_input': m.quantize_input.running_zero_point.flatten().tolist(),
                        
                        'stepsize': { 
                            'all': 1.0,
                        },
                        'scale_input': { 
                            'all': 1.0,
                        },
                        'radix': radix.flatten().tolist(),
                        'radix_input':radix_input.flatten().tolist(),
                        'bitwidth': {
                            'all': m.quantize_weight.num_bits,                        
                        },                        
                    }
        qparams = get_quantized_model_and_params(m, qparams)

    model.quantized = None
    return qparams

