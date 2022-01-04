from collections import namedtuple
import torch
import torch.nn as nn
from models.modules.quantize import calculate_qparams, quantize, QConv2d,QLinear

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def quantize_tensor(x, num_bits=8):
    qmin = -(2.**(num_bits-1) - 1.)
    qmax = 2.**(num_bits-1) - 1.
    abs_max_val = max(abs(x.min()), abs(x.max()))
    
    scale = abs_max_val / qmax
    zero_point = 0

    q_x = x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def dequantize_tensor(q_x):
    return q_x.scale * q_x.tensor.float() 


def quantize_model(model):
    qparams = {}

    for n, p in model.state_dict().items():
        qp = quantize_tensor(p)
        qparams[n + '.quantization.scale'] = torch.FloatTensor([qp.scale])
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
                         scale=params[n + '.quantization.scale'][0],
                         zero_point=params[n + '.quantization.zero_point'][0])
            p.copy_(dequantize_tensor(qp))
            model.register_buffer(n + '.quantization.scale', None)
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
                dqw = tensor_fl2fx2fl(m.weight, num_bits=m.quantize_weight.num_bits)
                m.weight.copy_(dqw)
                # qmax = 2.**m.quantize_weight.num_bits -1.
                qmax = 2.**(m.quantize_weight.num_bits-1) - 1.
                scale = m.quantize_weight.running_range / qmax
                qparams[m.name] = {
                        'shape': list(m.weight.shape),
                        
                        'num_bits': m.quantize_weight.num_bits,
                        'range': m.quantize_weight.running_range.flatten().tolist(),
                        'zero_point': m.quantize_weight.running_zero_point.flatten().tolist(),
                        
                        'num_bits_input': m.quantize_input.num_bits,
                        'range_input': m.quantize_input.running_range.flatten().tolist(),
                        'zero_point_input': m.quantize_input.running_zero_point.flatten().tolist(),
                        
                        'scale': { 
                            'all': scale.flatten().tolist(),
                        },
                        'radix': {
                            'all': 0,
                        },
                        'bitwidth': {
                            'all': m.quantize_weight.num_bits,                        
                        },                        
                    }
        qparams = get_quantized_model_and_params(m, qparams)

    model.quantized = None
    return qparams

