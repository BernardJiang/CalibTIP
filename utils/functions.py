import torch
from torch.autograd.function import Function

class ScaleGrad(Function):

    @staticmethod
    def forward(ctx, input, stepsize):
        ctx.stepsize = stepsize
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = ctx.stepsize * grad_output
        return grad_input, None


def scale_grad(x, stepsize):
    return ScaleGrad().apply(x, stepsize)

def negate_grad(x):
    return scale_grad(x, -1)
