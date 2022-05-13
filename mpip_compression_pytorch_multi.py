import pandas as pd
from pulp import *
import numpy as np
import argparse
from main import main_with_args as main_per_layer
import os
from itertools import zip_longest


Debug = False
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def mpip_compression(files=None, replace_precisions=None, Degradation=None, noise=None, method='acc', base_precision=8):
    data = {}
    if files[0] == '':
        files = files[1:]

    for f, prec in zip(files, replace_precisions):
        data[prec] = pd.read_csv(f)

    if Degradation is None:
        Degradation = 0.18

    bops=False
    metric = 'MACs' if bops else 'Parameters Size [Elements]'

    if method=='acc':
        acc=True
    elif method=='loss':
        acc=False
    measurement = 'accuracy' if acc else 'loss'

    po = 2 if bops else 1
    prob = LpProblem('BitAllocationProblem',LpMinimize)
    Combinations={}; accLoss={}; memorySaved={}; Indicators={}; S={}; DeltaL={}
    
    replace_precision = replace_precisions[0]  # 0 weight, 1 activation
    num_layers = len(data[replace_precision]['base precision']) - 1

    base_accuracy = data[replace_precision][measurement][0]
    total_mac=0
    for l in range(1,num_layers+1):
        layer = data[replace_precision]['replaced layer'][l]
        total_mac+= int(data[replace_precision][metric][l])
        base_performance = int(data[replace_precision][metric][l]) * (base_precision ** po)
        acc_layer = {}
        performance = {}
        Combinations[layer] = []
        accLoss[layer] = {}
        memorySaved[layer] = {}
        for prec_w, prec_a in grouper(2, replace_precisions):
            layer_w_a = layer + '_{}W_{}A'.format(prec_w, prec_a)  # 4w8a or 8w8a
            acc_layer[prec_w] = data[prec_w][measurement][l]
            performance[prec_w] = int(data[prec_w][metric][l]) * (prec_w ** po)
            Combinations[layer].append(layer_w_a)
            if acc:
                accLoss[layer][layer_w_a] = max(base_accuracy - acc_layer[prec_w], 1e-6)
            else:
                accLoss[layer][layer_w_a] = max(acc_layer[prec_w] - base_accuracy, 1e-6)
            if noise is not None:
                accLoss[layer][layer_w_a] += noise * np.random.normal() * accLoss[layer][layer_w_a]
            memorySaved[layer][layer_w_a] = base_performance - performance[prec_w]
        layer_BW_BA = layer + '_{}W_{}A'.format(base_precision, base_precision) # 8W8A
        Combinations[layer].append(layer_BW_BA)
        accLoss[layer][layer_BW_BA] = 0
        memorySaved[layer][layer_BW_BA] = 0
        Indicators[layer] = LpVariable.dicts("indicator"+layer,Combinations[layer],0,1,LpInteger)
        S[layer] =LpVariable("S"+layer, 0)
        DeltaL[layer] =LpVariable("DeltaL"+layer, 0)

    prob += lpSum([S[layer] for layer in S.keys()]) # Objective (minimize acc loss)

    total_performance=total_mac*base_precision**po

    for l in range(1,num_layers+1): # range(1,3):#
        layer = data[replace_precisions[0]]['replaced layer'][l]
        prob += lpSum([Indicators[layer][i] * accLoss[layer][i] for i in Combinations[layer]]) == S[layer]  # Accuracy loss per layer
        prob += lpSum([Indicators[layer][i] for i in Combinations[layer]]) == 1  # Constraint of only one indicator==1
        prob += lpSum([Indicators[layer][i] * memorySaved[layer][i] for i in Combinations[layer]]) == DeltaL[layer]  # Acc loss per layer

    prob += lpSum([DeltaL[layer] for layer in DeltaL.keys()]) >= total_performance*(1- Degradation*(32/base_precision)) # Total acc loss constraint

    prob.solve()
    LpStatus[prob.status]

    print('optimal solution for total degradation D = ' + str(Degradation)+':')
    if Debug:
        for v in prob.variables():
            print(v.name, "=", v.varValue)

        print(value(prob.objective))

    if (prob.status==-1):
        print('Infeasable')

    expected_acc_deg = sum([S[layer].varValue for layer in S.keys()])
    reduced_performance=sum([DeltaL[layer].varValue for layer in DeltaL.keys()])

    sol = {}
    memory_reduced = 0
    acc_deg = 0
    policy = []
    all_precisions = replace_precisions + [base_precision, base_precision]
    total_params = {}
    for prec_w, prec_a in grouper(2, all_precisions):
            total_params[prec_w] = 0
    for l in range(1, num_layers + 1):
        layer = data[replace_precisions[0]]['replaced layer'][l]
        for prec_w, prec_a in grouper(2, all_precisions):
            layer_w_a = layer + '_{}W_{}A'.format(prec_w, prec_a)
            if Indicators[layer][layer_w_a].varValue:
                policy.append('w{}a{}'.format(prec_w, prec_a))
                sol[layer] = [prec_w, prec_a]
                memory_reduced += memorySaved[layer][layer_w_a]
                acc_deg += accLoss[layer][layer_w_a]
                total_params[prec_w] += int(data[replace_precisions[0]][metric][l])

    print('Final Solution: ', sol)
    print('Policy: ', policy)
    print('Achieved compression: ', (total_performance - memory_reduced) / (total_performance * (32/base_precision)))
    if acc:
        expected_acc = base_accuracy - acc_deg
    else:
        expected_acc = base_accuracy + acc_deg
    print('Expected acc: ', expected_acc)
    for prec_w, prec_a in grouper(2, all_precisions):
        print('Params % in int {} = {}'.format(prec_w, total_params[prec_w] / total_mac))

    return sol, expected_acc, (total_performance - reduced_performance) / (total_performance * (32/base_precision)), policy

def mpip_compression2(files=None, replace_precisions=None, Degradation=None, noise=None, method='acc', base_precision=8, bops=False):
    data = {}
    if files[0] == '':
        files = files[1:]

    for f, prec in zip(files, replace_precisions):
        data[prec] = pd.read_csv(f)

    if Degradation is None:
        Degradation = 0.18

    # bops=False
    metric = 'MACs' if bops else 'Parameters Size [Elements]'

    if method=='acc':
        acc=True
    elif method=='loss':
        acc=False
    measurement = 'accuracy' if acc else 'loss'

    po = 2 if bops else 1
    prob = LpProblem('BitAllocationProblem',LpMinimize)
    Combinations={}; accLoss={}; memorySaved={}; Indicators={}; S={}; DeltaL={}
    
    replace_precision = replace_precisions[0]  # 0 weight, 1 activation
    num_layers = len(data[replace_precision]['base precision']) - 1

    base_accuracy = data[replace_precision][measurement][0]
    total_mac=0
    loss2gain = {}
    gains = {}
    for l in range(1,num_layers+1):
        layer = data[replace_precision]['replaced layer'][l]
        total_mac+= int(data[replace_precision][metric][l])
        base_performance = int(data[replace_precision][metric][l]) * ( base_precision * base_precision if bops else (base_precision ** po))
        acc_layer = {}
        performance = {}
        Combinations[layer] = []
        accLoss[layer] = {}
        memorySaved[layer] = {}
        loss2gain[layer] = 0
        
        # for prec_w, prec_a in grouper(2, replace_precisions):
        prec_w = replace_precisions[0]
        prec_a = replace_precisions[1]
        layer_w_a = layer + '_{}W_{}A'.format(prec_w, prec_a)  # 4w8a or 8w8a
        acc_layer[prec_w] = data[prec_w][measurement][l]
        performance[prec_w] = int(data[prec_w][metric][l]) * (prec_w * prec_a if bops else (prec_w ** po))
        Combinations[layer].append(layer_w_a)
        if acc:
            accLoss[layer][layer_w_a] = max(base_accuracy - acc_layer[prec_w], 1e-6)
        else:
            accLoss[layer][layer_w_a] = max(acc_layer[prec_w] - base_accuracy, 1e-6)
        if noise is not None:
            accLoss[layer][layer_w_a] += noise * np.random.normal() * accLoss[layer][layer_w_a]
        
        memorySaved[layer][layer_w_a] = max(base_performance - performance[prec_w], 1)
        
        loss2gain[layer] = accLoss[layer][layer_w_a] / memorySaved[layer][layer_w_a]
        gains[layer] = memorySaved[layer][layer_w_a]
        
        
        layer_BW_BA = layer + '_{}W_{}A'.format(base_precision, base_precision) # 8W8A
        Combinations[layer].append(layer_BW_BA)
        accLoss[layer][layer_BW_BA] = 0
        memorySaved[layer][layer_BW_BA] = 0

    
    #sort the loss2gain    
    sorted_loss2gain = {}
    weight_choices = {}
    sorted_keys = sorted(loss2gain, key=loss2gain.get)
    for w in sorted_keys:
        sorted_loss2gain[w]=loss2gain[w]
        weight_choices[w] = 8
    
    #search for K.
    K = 0
    gain_lower = 0
    gain_higher = 0
    total_performance = total_mac * ( base_precision * base_precision if bops else (base_precision ** po))  # Macs or bits
    #degradation = {0.13, 0.25} from maximum compression 48% to no compression 0%.
    target_gain = total_performance*(1- Degradation*(32/base_precision))
    for l in sorted_loss2gain.keys():
        gain_higher += gains[l]
        if gain_lower <= target_gain and target_gain < gain_higher:
            #TODO: found it.
            break
        gain_lower = gain_higher
        weight_choices[l] = 4
        K += 1
    
    print("Found K =", K)    
    
    reduced_performance = 0
    sol = {}
    memory_reduced = gain_lower
    acc_deg = 0
    policy = []
    all_precisions = replace_precisions + [base_precision, base_precision]
    total_params = {}
    for prec_w, prec_a in grouper(2, all_precisions):
            total_params[prec_w] = 0
    for l in range(1, num_layers + 1):
        layer = data[replace_precisions[0]]['replaced layer'][l]        
        if weight_choices[layer] == 4:
            prec_w = 4
            prec_a = 8
        else:
            prec_w = 8
            prec_a = 8
        
        layer_w_a = layer + '_{}W_{}A'.format(prec_w, prec_a)
        policy.append('w{}a{}'.format(prec_w, prec_a)) 
        sol[layer] = [prec_w, prec_a]
        if weight_choices[layer] == 4:
            # memory_reduced += performance[prec_w]
            acc_deg += accLoss[layer][layer_w_a]
        total_params[prec_w] += int(data[replace_precisions[0]][metric][l])
            
    comp = (total_performance - memory_reduced) / (total_performance * (32/base_precision)) 
    print('Final Solution: ', sol)
    print('Policy: ', policy)
    print('Achieved compression: ', comp)
    if acc:
        expected_acc = base_accuracy - acc_deg
    else:
        expected_acc = base_accuracy + acc_deg
    print('Expected acc: ', expected_acc)
    for prec_w, prec_a in grouper(2, all_precisions):
        print('Params % in int {} = {}'.format(prec_w, total_params[prec_w] / total_mac))

    return sol, expected_acc, comp, policy

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning')

    parser.add_argument('--device-ids', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3')
    parser.add_argument('--ip_method', type=str, default='loss', help='IP optimization target, loss / acc')
    parser.add_argument('--ip_gain', type=bool, default=False, help='IP optimization gain, weight / MACs')
    parser.add_argument('--model', type=str, default='resnet', help='model to use')
    parser.add_argument('--model_vis', type=str, default='resnet50', help='torchvision model name')
    parser.add_argument('--num_exp', default=1, type=int, help='number of experiments per compression level')
    parser.add_argument('--sigma', default=None, type=float, help='sigma noise to add to measurements')
    parser.add_argument('--layer_by_layer_files', type=str, default='./results/resnet50_w8a8_adaquant/resnet.absorb_bn.measure.adaquant.per_layer_accuracy.csv', help='layer degradation csv file')
    parser.add_argument('--datasets-dir', type=str, default='/media/drive/Datasets', help='dataset dir')
    parser.add_argument('--precisions', type=str, default='8;4', help='precisions, base first, separated by ;')
    parser.add_argument('--max_compression', type=float, default='0.25', help='max compression to test')
    parser.add_argument('--min_compression', type=float, default='0.13', help='min compression to test')
    parser.add_argument('--suffix', type=str, default='', help='suffix to add to all outputs')
    parser.add_argument('--do_not_use_adaquant', action='store_true', default=False,
                        help='use non optimized model')
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='evaluate on calibration data')
    parser.add_argument('--layers_precision_json_4_IP', type=str, default=None, help='json file from knerex to use')
    parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='/workspace/develop/CalibTIP/results',
                    help='results dir')

    args = parser.parse_args()
    return args


args = get_args()

compressions = np.arange(args.min_compression, args.max_compression, 0.01)
sigma = args.sigma
num_exp = args.num_exp
ip_method = args.ip_method
files = args.layer_by_layer_files.split(';')
precisions = [int(i) for i in args.precisions.split(';')]
replace_precisions = precisions[2:]
datasets_dir = args.datasets_dir
model = args.model
model_vis = args.model_vis
if args.do_not_use_adaquant:
    workdirs = [os.path.join(args.results_dir, model_vis + '_w{}a{}'.format(w1, a1)) for w1,a1 in grouper(2, precisions)]
else:
    # workdirs = [os.path.join(args.results_dir, model_vis + '_w{}a{}.adaquant'.format(w1, a1)) for w1,a1 in grouper(2, precisions)]
    workdirs = os.path.join(args.results_dir, model_vis + '.adaquant')
eval_dir = os.path.join(workdirs, model + '.absorb_bn')

perC=True
num_sp_layers=0
model_config = {'batch_norm': False,'measure': False, 'perC': perC}
if model_vis=='resnet18':
    model_config['depth'] = 18

output_fname = os.path.join(workdirs, 'IP_{}_{}{}.txt'.format(model_vis, ip_method, args.suffix))

eval_dict = {'model': model,
             'evaluate': eval_dir,
             'dataset': 'imagenet_calib',
             'datasets_dir': datasets_dir,
             'b': 100,
             'model_config': model_config,
             'mixed_builder': True,
             'device_ids': args.device_ids,
             'precisions': precisions,
             'layers_precision_json_4_IP': args.layers_precision_json_4_IP
             }

if args.do_not_use_adaquant:
    eval_dict['opt_model_paths'] = [os.path.join(dd, model + '.absorb_bn.measure_perC') for dd in workdirs]
else:
    eval_dict['opt_model_paths'] = [os.path.join(workdirs, model + '.absorb_bn.w{}a{}.measure_perC.adaquant'.format(w1,a1)) for w1,a1 in grouper(2, precisions)]

if args.eval_on_train:
    eval_dict['eval_on_train'] = True

solutions = []
expected_accuracies = []
state_dict_path=[]
actual_compressions = []
actual_accuracies = []
actual_losses = []
policies = []
completed = 0
start_from = 0
for Deg in compressions:
    if completed < start_from:
        completed += 1
        solutions.append('')
        state_dict_path.append('')
        policies.append([])
        expected_accuracies.append(0)
        actual_compressions.append(0)
        actual_accuracies.append(0)
        actual_losses.append(0)
        continue
    attempted_policies = {}
    valid_exp = 0
    while valid_exp < num_exp:
        if Debug:
            import pdb; pdb.set_trace()
        sol, expect_acc, comp, policy = mpip_compression2(files=files, replace_precisions=replace_precisions, Degradation=Deg, noise=sigma, method=ip_method, bops=args.ip_gain)
        if str(policy) in attempted_policies.keys():
            continue
        valid_exp += 1

        eval_dict['names_sp_layers'] = sol
        eval_dict['suffix'] = 'comp_{}_{}{}'.format( "{:.2f}".format(Deg), ip_method, args.suffix)
        eval_dict['nbits_weight'] = replace_precisions[0]
        eval_dict['nbits_act'] = replace_precisions[1]
        acc, loss = main_per_layer(**eval_dict)
        # acc = 0.11; loss = 0.9
        # import pdb; pdb.set_trace()

        attempted_policies[str(policy)] = acc

        solutions.append(sol)
        policies.append(policy.copy())
        expected_accuracies.append(expect_acc)
        actual_compressions.append(comp)
        actual_accuracies.append(acc)
        actual_losses.append(loss)
        state_dict_path.append(eval_dict['evaluate']+'.mixed-ip-results.'+eval_dict['suffix'])
        completed += 1
        c = 0
        for d in compressions:
            for exp in range(num_exp):
                if c >= completed:
                    break
                print('Compression thr {}, experiment {},state_dict_path {}, compression {}, expected {} {}, actual acc {}, actual loss {}'.format("{:.2f}".format(d), exp, state_dict_path[c], actual_compressions[c],
                                                                                            ip_method, expected_accuracies[c],
                                                                                            actual_accuracies[c], actual_losses[c]))
                print('Policy: {}'.format(policies[c]))
                print('Configuration = {}'.format(solutions[c]))
                c += 1


with open(output_fname, 'w') as pid:
    line = 'Compression thr\tExperiment\tstate_dict_path\tActual compression\tExpected {}\tActual Accuracy\tActual loss\tPolicy\tConfiguration\n'.format(ip_method)
    pid.write(line)
    c = 0
    for Deg in compressions:
        for exp in range(num_exp):
            print('Compression thr {}, experiment {}, state_dict_path {}, actual_compression {}, expected {} {}, actual acc {}, actual loss {}'.format(Deg, exp,state_dict_path[c], actual_compressions[c], ip_method, expected_accuracies[c], actual_accuracies[c], actual_losses[c]))
            print('Policy: {}'.format(policies[c]))
            line = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format( "{:.2f}".format(Deg), exp, state_dict_path[c], actual_compressions[c], expected_accuracies[c], actual_accuracies[c], actual_losses[c], policies[c], solutions[c])
            pid.write(line)
            c += 1
