import torch
from torch.autograd.function import InplaceFunction, Function
class Clamp(InplaceFunction):
    def forward(ctx, input,inplace, min, max):
        ctx.inplace = inplace                                                                          
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        output.clamp_(min, max)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input,None, None, None

a = torch.tensor([-2., -1., 0., 1., 2])
a.requires_grad = True
b = torch.clamp(a, -1., 1.)
b.retain_grad()
c = b.mean()
c.backward()
print("Internal   Clamp(): ", b.grad, a.grad)

d = torch.tensor([-2., -1., 0., 1., 2])
d.requires_grad = True
e = Clamp().apply(d, False, -1., 1.)
e.retain_grad()
f = e.mean()
f.backward()
print("Customized Clamp(): ", e.grad, d.grad)
