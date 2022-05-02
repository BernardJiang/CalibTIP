import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import scipy.optimize as opt
import math
from .log import get_linenumber, get_gpu_memory_map, check_memory_usage
import copy
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('/workspace/develop/pytorch-lamb/pytorch_lamb')
from pytorch_lamb import Lamb, log_lamb_rs

def optimize_qparams(layer, cached_inps, cached_outs, test_inp, test_out, batch_size=100):
    print("\nOptimize quantization params")
    # w_range_orig = layer.quantize_weight.running_range.data.clone()
    # w_zp_orig = layer.quantize_weight.running_zero_point.data.clone()
    # inp_range_orig = layer.quantize_input.running_range.data.clone()
    # inp_zp_orig = layer.quantize_input.running_zero_point.data.clone()

    def layer_err(p, inp, out):
        # layer.quantize_weight.running_range.data = w_range_orig * p[0]
        # layer.quantize_weight.running_zero_point.data = w_zp_orig + p[1]
        # layer.quantize_input.running_range.data = inp_range_orig * p[2]
        # layer.quantize_input.running_zero_point.data = inp_zp_orig + p[3]
        yq = layer(inp)
        return F.mse_loss(yq, out).item()

    init = np.array([1, 0, 1, 0])
    results = []
    for i in tqdm(range(int(cached_inps.size(0) / batch_size))):
        cur_inp = cached_inps[i * batch_size:(i + 1) * batch_size]
        cur_out = cached_outs[i * batch_size:(i + 1) * batch_size]

        # print("init:")
        # print(init)
        res = opt.minimize(lambda p: layer_err(p, cur_inp, cur_out), init, method=methods[0])
        results.append(res.x)

    mean_res = np.array(results).mean(axis=0)
    print(mean_res)
    mse_before = layer_err(init, test_inp, test_out)
    mse_after = layer_err(mean_res, test_inp, test_out)
    return mse_before, mse_after


def adaquant(layer, cached_inps, cached_outs, test_inp, test_out, lr1=1e-4, lr2=1e-2, iters=100, progress=True, batch_size=64, relu=False, writer=None):
    # print("\nRun adaquant")

    # Writer will output to ./runs/ directory by default
    # writer = SummaryWriter()
    
    # get_gpu_memory_map()
    # check_memory_usage()

    # with torch.no_grad():
    #     oldweights = copy.deepcopy(layer.weight)
    #     if hasattr(layer, 'bias') and layer.bias is not None: 
    #         oldbias = copy.deepcopy(layer.bias)    
            
    if relu:
        mse_before = F.mse_loss(F.relu_(layer(test_inp)), F.relu_(test_out))
    else:
        mse_before = F.mse_loss(layer(test_inp), test_out)

    # lr_factor = 1e-2
    # Those hyperparameters tuned for 8 bit and checked on mobilenet_v2 and resnet50
    # Have to verify on other bit-width and other models
    lr_qpin = 1e-2 #lr_factor * (test_inp.max() - test_inp.min()).item()  # 1e-1
    lr_qpw = 1e-2 #lr_factor * (layer.weight.max() - layer.weight.min()).item()  # 1e-3
    lr_w = 1e-5 # mse_before.cpu().detach().numpy() # 1e-5 # 0.0025 # 1e-6 #lr_factor * layer.weight.std().item()  # 1e-5
    lr_b = lr_w # mse_before.cpu().numpy() # 1e-5 # 0.0025 # 1e-6#lr_factor * layer.bias.std().item()  # 1e-3
    weight_decay = 0.01
    if layer.quantize_weight.qmax == 127:
        lr_qpin /= 10.
        lr_qpw /= 10. 
        print("8-bit layer should use smaller LR =", lr_qpw )
        
    opt_w = Lamb([layer.weight], lr=lr_w, weight_decay=weight_decay, betas=(.9, .999), adam=True)

    # scheduler_w = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_w,
                                                        #  min_lr=1e-8,
                                                        #  factor=0.9,
                                                        #  verbose=False,
                                                        #  patience=10)
    
    if hasattr(layer, 'bias') and layer.bias is not None: 
        opt_bias = Lamb([layer.bias], lr=lr_b, weight_decay=weight_decay, betas=(.9, .999), adam=True)
        # scheduler_bias = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_bias,
                                                        #  min_lr=1e-8,
                                                        #  factor=0.9,
                                                        #  verbose=False,
                                                        #  patience=10)
        
    # opt_w = torch.optim.AdamW([layer.weight], lr=lr_w)
    # if hasattr(layer, 'bias') and layer.bias is not None: opt_bias = torch.optim.AdamW([layer.bias], lr=lr_b)
    
    # opt_qparams_in = torch.optim.Adam([layer.quantize_input.running_range,
    #                                    layer.quantize_input.running_zero_point], lr=lr_qpin)
    # opt_qparams_w = torch.optim.Adam([layer.quantize_weight.running_range,
    #                                   layer.quantize_weight.running_zero_point], lr=lr_qpw)
    
    opt_in_scale = Lamb([layer.quantize_input.running_scale], lr=lr_qpin)
    opt_out_scale = Lamb([layer.quantize_weight.running_scale], lr=lr_qpw)                      
    # scheduler_in_scale = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_in_scale,
                                                        #  min_lr=1e-8,
                                                        #  factor=0.9,
                                                        #  verbose=False,
                                                        #  patience=10)
    # scheduler_out_scale = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_out_scale,
                                                        #  min_lr=1e-8,
                                                        #  factor=0.9,
                                                        #  verbose=False,
                                                        #  patience=10)
 

    if writer is not None:
        writer.add_scalar("layer/{}".format(layer.name), mse_before.item(), 0)

    losses = []
    for j in (tqdm(range(iters)) if progress else range(iters)):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]

        # print("idx: ", j)
        # print(__file__, get_linenumber())
        # get_gpu_memory_map()

        train_inp = cached_inps[idx]#.cuda()
        train_out = cached_outs[idx]#.cuda()

        qout = layer(train_inp)
        if relu:
            loss = F.mse_loss(F.relu_(qout), F.relu_(train_out))
        else:    
            loss = F.mse_loss(qout, train_out)

        if writer is not None:
            if j % 10 == 0 :
                writer.add_scalar("layer/{}".format(layer.name), loss.item(), j)
                writer.add_scalar("layer/{}output_scale[0]".format(layer.name), layer.quantize_weight.running_scale[0], j)
                writer.add_scalar("layer/{}input_scale[0]".format(layer.name), layer.quantize_input.running_scale[0], j)

        losses.append(loss.item())
        opt_w.zero_grad()
        if hasattr(layer, 'bias') and layer.bias is not None: 
            opt_bias.zero_grad()

        opt_in_scale.zero_grad()
        opt_out_scale.zero_grad()
        loss.backward()
        opt_w.step()
        # scheduler_w.step(loss)
        if hasattr(layer, 'bias') and layer.bias is not None: 
            opt_bias.step()
            # scheduler_bias.step(loss)
        opt_in_scale.step()
        opt_out_scale.step()
        # scheduler_in_scale.step(loss)
        # scheduler_out_scale.step(loss)
        
        # if layer.name == 'conv1':
        #     print("iter {}, in range/zp {} {}, w range/zp {} {} ".format(j, layer.quantize_input.running_range.item(), layer.quantize_input.running_zero_point.item(), layer.quantize_weight.running_range[0].item(), layer.quantize_weight.running_zero_point[0].item()))

            # if len(losses) < 10:
            #     total_loss = loss.item()
            # else:
            #     total_loss = np.mean(losses[-10:])
            # print("mse out: {}, pc mean loss: {}, total: {}".format(mse_out.item(), mean_loss.item(), total_loss))

    if relu:
        mse_after = F.mse_loss(F.relu_(layer(test_inp)), F.relu_(test_out))
    else:
        mse_after = F.mse_loss(layer(test_inp), test_out)

    # if mse_before.item() < mse_after.item():        
    #     print("Bernard revert adaquant for this layer!")
    #     with torch.no_grad():
    #         layer.weight.copy_(oldweights)
    #         if hasattr(layer, 'bias') and layer.bias is not None: 
    #             layer.bias.copy_(oldbias)
    
    if writer is not None:
         writer.add_scalar("layer/{}".format(layer.name), mse_after.item(), iters-1)
         writer.add_scalar("layer/{}output_scale[0]".format(layer.name), layer.quantize_weight.running_scale[0], iters-1)
         writer.add_scalar("layer/{}input_scale[0]".format(layer.name), layer.quantize_input.running_scale[0], iters-1)

    
    return mse_before.item(), mse_after.item()


def optimize_layer(layer, in_out, optimize_weights=False, batch_size=100, model_name=None, writer=None):

    cached_inps = torch.cat([x[0] for x in in_out]).to(layer.weight.device)
    cached_outs = torch.cat([x[1] for x in in_out]).to(layer.weight.device)

    idx = torch.randperm(cached_inps.size(0))[:batch_size]

    test_inp = cached_inps[idx]
    test_out = cached_outs[idx]

    # mse_before, mse_after = optimize_qparams(layer, cached_inps, cached_outs, test_inp, test_out)
    # mse_before_opt = mse_before
    # print("MSE before qparams: {}".format(mse_before))
    # print("MSE after qparams: {}".format(mse_after))

    if optimize_weights:
        relu_condition = lambda layer_name: False
        if model_name is not None and 'resnet' in model_name:
            relu_condition = lambda layer_name: 'conv1' in layer_name or 'conv2' in layer_name
        elif 'inception_v3' == model_name:
            relu_condition = lambda layer_name: 'conv' in layer_name
        elif 'mobilenet_v2' == model_name:
            relu_condition = lambda layer_name: layer_name.endswith('0.0') or layer_name.endswith('1.0') or layer_name.endswith('18.0')

        # print(__file__, get_linenumber())
        # get_gpu_memory_map()
        # check_memory_usage()
        relu_flag = relu_condition(layer.name)      
        mse_before, mse_after = adaquant(layer, cached_inps, cached_outs, test_inp, test_out, iters=300, batch_size=batch_size, lr1=1e-5, lr2=1e-4, relu=relu_flag, writer=writer) 
        mse_before_opt = mse_before
        print("\nMSE before adaquant: {:e}".format(mse_before))
        print("MSE after  adaquant: {:e}".format(mse_after))
        torch.cuda.empty_cache()
    else:
        mse_before, mse_after = optimize_qparams(layer, cached_inps, cached_outs, test_inp, test_out)
        mse_before_opt = mse_before
        print("MSE before qparams: {}".format(mse_before))
        print("MSE after qparams: {}".format(mse_after))

    mse_after_opt = mse_after

    with torch.no_grad():
        N = test_out.numel()
        snr_before = (1/math.sqrt(N)) * math.sqrt(N * mse_before_opt) / torch.norm(test_out).item()
        snr_after = (1/math.sqrt(N)) * math.sqrt(N * mse_after_opt) / torch.norm(test_out).item()

    # optimize_rounding(layer, cached_inps, cached_outs, test_inp, test_out, iters=7000)
    # optimize_qparams(layer, cached_inps, cached_outs, test_inp, test_out)
    # optimize_rounding(layer, cached_inps, cached_outs, test_inp, test_out, iters=2000)
    # optimize_qparams(layer, cached_inps, cached_outs, test_inp, test_out)
    # optimize_rounding(layer, cached_inps, cached_outs, test_inp, test_out, iters=2000)
    # optimize_qparams(layer, test_inp, test_out)

    kurt_in = kurtosis(test_inp).item()
    kurt_w = kurtosis(layer.weight).item()

    del cached_inps
    del cached_outs
    torch.cuda.empty_cache()

    return mse_before_opt, mse_after_opt, snr_before, snr_after, kurt_in, kurt_w


def kurtosis(x):
    var = torch.mean((x - x.mean())**2)
    return torch.mean((x - x.mean())**4 / var**2)


def dump(model_name, layer, in_out):
    path = os.path.join("dump", model_name, layer.name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    if hasattr(layer, 'groups'):
        f = open(os.path.join(path, "groups_{}".format(layer.groups)), 'x')
        f.close()

    cached_inps = torch.cat([x[0] for x in in_out])
    cached_outs = torch.cat([x[1] for x in in_out])
    torch.save(cached_inps, os.path.join(path, "input.pt"))
    torch.save(cached_outs, os.path.join(path, "output.pt"))
    torch.save(layer.weight, os.path.join(path, 'weight.pt'))
    if layer.bias is not None:
        torch.save(layer.bias, os.path.join(path, 'bias.pt'))

