from collections import namedtuple
import torch
import torch.nn as nn

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def quantize_tensor(x, num_bits=8):
    qmin = -(2.**(num_bits-1) - 1.)
    qmax = 2.**(num_bits-1) - 1.
    min_val, max_val = x.min(), x.max()
    
    abs_max_val = abs(max_val)
    if abs(min_val) > abs_max_val:
        abs_max_val = abs(min_val)

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
