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

q = torch.tensor(1.)
q.requires_grad = True
a = torch.tensor([-2000., -2., -1., 0., 1., -2., 100.])
a.requires_grad = True
b = torch.clamp(a, -q, q)
b.retain_grad()
c = b.mean()
c.backward()
print("Internal   Clamp(): ", c, b.grad, a.grad, q.grad)

q2 = torch.tensor(1.)
q2.requires_grad = True
d = torch.tensor([-2000., -2., -1., 0., 1., -2., 100.])
d.requires_grad = True
e = Clamp().apply(d, False, -q2, q2)
e.retain_grad()
f = e.mean()
f.backward()
print("Customized Clamp(): ", f, e.grad, d.grad, q2.grad)
