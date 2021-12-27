from collections import namedtuple
import torch
import torch.nn as nn
from models.modules.quantize import calculate_qparams, quantize, QConv2d,QLinear

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def quantize_tensor(x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def tensor_fl2fx(qt, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    scale = qt.scale / (qmax - qmin) #x.scale is actual range 

    q_x = (qt.tensor - qt.zero_point) / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte().float()
    return q_x #QTensor(tensor=q_x, scale=scale, zero_point=qt.zero_point)


def tensor_fx2fl(qt, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    scale = qt.scale / (qmax - qmin) #x.scale is actual range 
    return scale * qt.tensor.float() + qt.zero_point

def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


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


def quantize_model_new(model, qparams = {}):    
    for i,m in enumerate(model.children()):
        if is_q_module(m):
            print("Found a quanted module m=", m.name, "weight = ", m.weight.shape, " bias ", m.bias.shape, " range =" , m.quantize_weight.running_range.shape, "zp =", m.quantize_weight.running_zero_point.shape, "numbits=", m.quantize_weight.num_bits)
            qp = QTensor(tensor=m.weight,
                         scale= m.quantize_weight.running_range,
                         zero_point=m.quantize_weight.running_zero_point)
            qw = tensor_fl2fx(qp, num_bits=m.quantize_weight.num_bits)
            with torch.no_grad():
                m.weight.copy_(qw)
                qparams[m.name] = {
                        'shape': list(m.weight.shape),
                        'range': m.quantize_weight.running_range.flatten().tolist(),
                        'zero_point': m.quantize_weight.running_zero_point.flatten().tolist(),
                        'num_bits': m.quantize_weight.num_bits,
                        'range_input': m.quantize_input.running_range.flatten().tolist(),
                        'zero_point_input': m.quantize_input.running_zero_point.flatten().tolist(),
                        'num_bits_input': m.quantize_input.num_bits
                    }
        qparams = quantize_model_new(m, qparams)

    model.quantized = None
    return qparams

def dequantize_model_new(model):
    for i,m in enumerate(model.children()):
        if is_q_module(m):
            print("Found a quanted module m=", m.name, "weight = ", m.weight.shape, " bias ", m.bias.shape, " range =" , m.quantize_weight.running_range.shape, "zp =", m.quantize_weight.running_zero_point.shape, "numbits=", m.quantize_weight.num_bits)
            qp = QTensor(tensor=m.weight,
                         scale= m.quantize_weight.running_range,
                         zero_point=m.quantize_weight.running_zero_point)
            qw = tensor_fx2fl(qp, num_bits=m.quantize_weight.num_bits)
            with torch.no_grad():
                m.weight.copy_(qw)
            # model.register_buffer(n + '.quantization.scale', None)
            # model.register_buffer(n + '.quantization.zero_point', None)
        dequantize_model_new(m)

    model.quantized = None
    return 
