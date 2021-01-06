import torch
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
def gather_bn_weights(module_list, prune_idx):

    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size

    return bn_weights

def write_cfg(cfg_file, module_defs):

    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key == 'batch_normalize' and value == 0:
                    continue

                if key != 'type':
                    if key == 'anchors':
                        value = ', '.join(','.join(str(int(i)) for i in j) for j in value)
                    f.write(f"{key}={value}\n")
            f.write("\n")
    return cfg_file



def parse_module_defs(module_defs):

    CBL_idx = []
    Conv_idx = []
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):

        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
            if module_defs[i+1]['type'] == 'maxpool' and module_defs[i+2]['type'] == 'route':
                #spp前一个CBL不剪 区分tiny
                ignore_idx.add(i)
            if module_defs[i+1]['type'] == 'route' and 'groups' in module_defs[i+1]:
                ignore_idx.add(i)

        elif module_def['type'] == 'shortcut':
            ignore_idx.add(i-1)
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                ignore_idx.add(identity_idx - 1)

        elif module_def['type'] == 'upsample':
            #上采样层前的卷积层不裁剪
            ignore_idx.add(i - 1)


    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx, ignore_idx


def parse_module_defs2(module_defs):
    CBL_idx = []
    Conv_idx = []
    shortcut_idx = dict()
    shortcut_all = set()
    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
            if module_defs[i + 1]['type'] == 'maxpool' and module_defs[i + 2]['type'] == 'route':
                # spp前一个CBL不剪 区分spp和tiny
                ignore_idx.add(i)
            if module_defs[i + 1]['type'] == 'route' and 'groups' in module_defs[i + 1]:
                ignore_idx.add(i)

        elif module_def['type'] == 'upsample':
            # 上采样层前的卷积层不裁剪
            ignore_idx.add(i - 1)

        elif module_def['type'] == 'shortcut':
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':

                # ignore_idx.add(identity_idx)
                shortcut_idx[i - 1] = identity_idx
                shortcut_all.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':

                # ignore_idx.add(identity_idx - 1)
                shortcut_idx[i - 1] = identity_idx - 1
                shortcut_all.add(identity_idx - 1)
            shortcut_all.add(i - 1)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx, shortcut_idx, shortcut_all

def get_input_mask(module_defs, idx, CBLidx2mask):

    if idx == 0:
        return np.ones(3)

    if module_defs[idx - 1]['type'] == 'convolutional':
        return CBLidx2mask[idx - 1]
    elif module_defs[idx - 1]['type'] == 'shortcut':
        return CBLidx2mask[idx - 2]
    elif module_defs[idx - 1]['type'] == 'route':
        route_in_idxs = []
        for layer_i in module_defs[idx - 1]['layers'].split(","):
            if int(layer_i) < 0:
                route_in_idxs.append(idx - 1 + int(layer_i))
            else:
                route_in_idxs.append(int(layer_i))

        if len(route_in_idxs) == 1:
            mask = CBLidx2mask[route_in_idxs[0]]
            if 'groups' in module_defs[idx - 1]:
                return mask[(mask.shape[0]//2):]
            return mask

        elif len(route_in_idxs) == 2:
            # return np.concatenate([CBLidx2mask[in_idx - 1] for in_idx in route_in_idxs])
            if module_defs[route_in_idxs[0]]['type'] == 'upsample':
                mask1 = CBLidx2mask[route_in_idxs[0] - 1]
            elif module_defs[route_in_idxs[0]]['type'] == 'convolutional':
                mask1 = CBLidx2mask[route_in_idxs[0]]
            if module_defs[route_in_idxs[1]]['type'] == 'convolutional':
                mask2 = CBLidx2mask[route_in_idxs[1]]
            else:
                mask2 = CBLidx2mask[route_in_idxs[1] - 1]
            return np.concatenate([mask1, mask2])

        elif len(route_in_idxs) == 4:
            #spp结构中最后一个route
            mask = CBLidx2mask[route_in_idxs[-1]]
            return np.concatenate([mask, mask, mask, mask])

        else:
            print("Something wrong with route module!")
            raise Exception
    elif module_defs[idx - 1]['type'] == 'maxpool':  #tiny
        if module_defs[idx - 2]['type'] == 'route':  #v4 tiny
            return get_input_mask(module_defs, idx - 1, CBLidx2mask)
        else:
            return CBLidx2mask[idx - 2]  #v3 tiny


def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):

    for idx in CBL_idx:
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

        compact_bn, loose_bn         = compact_CBL[1], loose_CBL[1]
        compact_bn.weight.data       = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data         = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data  = loose_bn.running_var.data[out_channel_idx].clone()

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data   = loose_conv.bias.data.clone()



class BNOptimizer():

    @staticmethod
    def updateBN(sr_flag, module_list, s, prune_idx, idx2mask=None):
        if sr_flag:
            # s = s if epoch <= opt.epochs * 0.5 else s * 0.01
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                bn_module = module_list[idx][1]
                bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1
            if idx2mask:
                for idx in idx2mask:
                    bn_module = module_list[idx][1]
                    #bn_module.weight.grad.data.add_(0.5 * s * torch.sign(bn_module.weight.data) * (1 - idx2mask[idx].cuda()))
                    bn_module.weight.grad.data.sub_(0.99 * s * torch.sign(bn_module.weight.data) * idx2mask[idx].cuda())


def merge_mask(model, CBLidx2mask, CBLidx2filters):
    for i in range(len(model.module_defs) - 1, -1, -1):
        mtype = model.module_defs[i]['type']
        if mtype == 'shortcut':
            if model.module_defs[i]['is_access']:
                continue

            Merge_masks = []
            layer_i = i
            while mtype == 'shortcut':
                model.module_defs[layer_i]['is_access'] = True

                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                    if bn:
                        Merge_masks.append(CBLidx2mask[layer_i - 1].unsqueeze(0))

                layer_i = int(model.module_defs[layer_i]['from']) + layer_i
                mtype = model.module_defs[layer_i]['type']

                if mtype == 'convolutional':
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        Merge_masks.append(CBLidx2mask[layer_i].unsqueeze(0))

            if len(Merge_masks) > 1:
                Merge_masks = torch.cat(Merge_masks, 0)
                merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float()
            else:
                merge_mask = Merge_masks[0].float()

            layer_i = i
            mtype = 'shortcut'
            while mtype == 'shortcut':

                if model.module_defs[layer_i - 1]['type'] == 'convolutional':
                    bn = int(model.module_defs[layer_i - 1]['batch_normalize'])
                    if bn:
                        CBLidx2mask[layer_i - 1] = merge_mask
                        CBLidx2filters[layer_i - 1] = int(torch.sum(merge_mask).item())

                layer_i = int(model.module_defs[layer_i]['from']) + layer_i
                mtype = model.module_defs[layer_i]['type']

                if mtype == 'convolutional':
                    bn = int(model.module_defs[layer_i]['batch_normalize'])
                    if bn:
                        CBLidx2mask[layer_i] = merge_mask
                        CBLidx2filters[layer_i] = int(torch.sum(merge_mask).item())

def update_activation(i, pruned_model, activation, CBL_idx):
    next_idx = i + 1
    if pruned_model.module_defs[next_idx]['type'] == 'convolutional':
        next_conv = pruned_model.module_list[next_idx][0]
        conv_sum = next_conv.weight.data.sum(dim=(2, 3))
        offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
        if next_idx in CBL_idx:
            next_bn = pruned_model.module_list[next_idx][1]
            next_bn.running_mean.data.sub_(offset)
        else:
            next_conv.bias.data.add_(offset)

def prune_model_keep_size2(model, prune_idx, CBL_idx, CBLidx2mask):
    pruned_model = deepcopy(model)
    activations = []
    for i, model_def in enumerate(model.module_defs):

        if model_def['type'] == 'convolutional':
            activation = torch.zeros(int(model_def['filters'])).cuda()
            if i in prune_idx:
                mask = torch.from_numpy(CBLidx2mask[i]).cuda()
                bn_module = pruned_model.module_list[i][1]
                bn_module.weight.data.mul_(mask)
                if model_def['activation'] == 'leaky':
                    activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)
                elif model_def['activation'] == 'mish':
                    activation = (1 - mask) * bn_module.bias.data.mul(F.softplus(bn_module.bias.data).tanh())
                update_activation(i, pruned_model, activation, CBL_idx)
                bn_module.bias.data.mul_(mask)
            activations.append(activation)

        elif model_def['type'] == 'shortcut':
            actv1 = activations[i - 1]
            from_layer = int(model_def['from'])
            actv2 = activations[i + from_layer]
            activation = actv1 + actv2
            update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)



        elif model_def['type'] == 'route':
            # spp不参与剪枝，其中的route不用更新，仅占位
            from_layers = [int(s) for s in model_def['layers'].split(',')]
            activation = None
            if len(from_layers) == 1:
                activation = activations[i + from_layers[0] if from_layers[0] < 0 else from_layers[0]]
                if 'groups' in model_def:
                    activation = activation[(activation.shape[0] // 2):]
                update_activation(i, pruned_model, activation, CBL_idx)
            elif len(from_layers) == 2:
                actv1 = activations[i + from_layers[0]]
                actv2 = activations[i + from_layers[1] if from_layers[1] < 0 else from_layers[1]]
                activation = torch.cat((actv1, actv2))
                update_activation(i, pruned_model, activation, CBL_idx)
            activations.append(activation)

        elif model_def['type'] == 'upsample':
            # activation = torch.zeros(int(model.module_defs[i - 1]['filters'])).cuda()
            activations.append(activations[i - 1])

        elif model_def['type'] == 'yolo':
            activations.append(None)

        elif model_def['type'] == 'maxpool':  # 区分spp和tiny
            if model.module_defs[i + 1]['type'] == 'route':
                activations.append(None)
            else:
                activation = activations[i - 1]
                update_activation(i, pruned_model, activation, CBL_idx)
                activations.append(activation)

    return pruned_model

def gather_l1_weights_channels(module_list, prune_idx):

    size_list = [module_list[idx][0].weight.data.shape[0]  for idx in prune_idx]
    l1_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        C, N, H, W = module_list[idx][0].weight.data.shape
        channels_weights = torch.reshape(module_list[idx][0].weight.data.abs(), (C, -1))
        channels_weights_sum = torch.sum(channels_weights, dim=1)/(N*H*W)
        # print('channels_weights_sum is', channels_weights_sum)
        # print('channels_weights_sum is', channels_weights_sum.shape)
        l1_weights[index:(index + size)] = channels_weights_sum
        index += size

    return l1_weights

def gather_l2_weights_channels(module_list, prune_idx):
    size_list = [module_list[idx][0].weight.data.shape[0] for idx in prune_idx]
    l2_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        C, N, H, W = module_list[idx][0].weight.data.shape
        # print("module_list[idx][0].weight.data.shape is", module_list[idx][0].weight.data.shape)
        channels_weights = module_list[idx][0].weight.data.view(module_list[idx][0].weight.data.size()[0], -1)
        channels_l2_weights = torch.norm(channels_weights, 2, 1)/(H*W*N)
        # channels_l2_weights_sum = torch.sum(channels_l2_weights, dim=-1)/(H*W*N)
        # print('channels_l2_weights_sum is', channels_l2_weights)
        l2_weights[index:(index + size)] = channels_l2_weights
        index += size
    return l2_weights

def get_global_norm_thr(model, prune_idx, global_percent, norm_type="l2"):
    if norm_type == "l1":
        norm_weights = gather_l1_weights_channels(model.module_list, prune_idx)
    elif norm_type== "l2":
        norm_weights = gather_l2_weights_channels(model.module_list, prune_idx)
    sorted_l2, sorted_l2_index = torch.sort(norm_weights)
    thresh_index_norm = int(len(norm_weights)*global_percent)
    thresh_norm = sorted_l2[thresh_index_norm].cuda()
    return thresh_norm

def get_layer_norm_thr(model, prune_idx, layer_percent, norm_type="l2"):
    size_list = [model.module_list[idx][0].weight.data.shape[0] for idx in prune_idx]
    layer_thr_list = []
    layer_prune_index = []
    for idx, size in zip(prune_idx, size_list):
        C, N, H, W = model.module_list[idx][0].weight.data.shape
        channels_weights = model.module_list[idx][0].weight.data.view(model.module_list[idx][0].weight.data.size()[0], -1)
        channels_l2_weights = torch.norm(channels_weights, 2, 1)
        thr_index = int(len(channels_l2_weights.cpu().numpy().tolist())*layer_percent)
        sort_weights, sort_index = torch.sort(channels_l2_weights)
        thr = sort_weights[thr_index]
        layer_thr_list.append(thr)
        layer_prune_index.append(sort_index[:thr_index])
        # print("thr_index is", sort_index[:thr_index])
        # raise ValueError("stop")
    return layer_thr_list, layer_prune_index

def obtain_filters_mask_norm(model, thre, CBL_idx, prune_idx, layer_keep, norm_type="l2"):
    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        if idx in prune_idx:
            channels, N, H, W = model.module_list[idx][0].weight.data.shape
            min_channel_num = int(channels * layer_keep) if int(channels * layer_keep) > 0 else 1
            # print("nb_of_clusters is", min_channel_num)
            if norm_type == "l1":
                weight_copy_sum = torch.sum(torch.reshape(model.module_list[idx][0].weight.data.abs(), (channels, -1)), dim=1) / (N * H * W)
            elif norm_type == "l2":
                weight_copy_sum = torch.norm(model.module_list[idx][0].weight.data.view(channels, -1), 2, 1) / ( H * W * N)
            else:
                raise ValueError("norm type is illegal")
            # print("weight_copy_sum is", weight_copy_sum)
            mask = weight_copy_sum.gt(thre).float()
            if int(torch.sum(mask)) < min_channel_num:
                _, sorted_index_weights = torch.sort(weight_copy_sum, descending=True)
                mask[sorted_index_weights[:min_channel_num]] = 1.
            # print("at last mask is", mask)
            remain = int(mask.sum())
            pruned = pruned + mask.shape[0] - remain

            print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                  f'remaining channel: {remain:>4d}')
        else:
            mask = torch.ones(bn_module.weight.data.shape)
            remain = mask.shape[0]
            print(f'not prune layer info:  layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                  f'remaining channel: {remain:>4d}')
        total += mask.shape[0]
        num_filters.append(remain)
        filters_mask.append(mask.clone())
    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')
    return num_filters, filters_mask


def obtain_filters_mask_norm_per_layer(model, thre_list, CBL_idx, prune_idx, layer_keep, norm_type="l2"):
    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    thre_id = 0
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        if idx in prune_idx:
            thre = thre_list[thre_id]
            thre_id += 1
            # print("this layer thr is", thre)
            channels, N, H, W = model.module_list[idx][0].weight.data.shape
            min_channel_num = int(channels * layer_keep) if int(channels * layer_keep) > 0 else 1
            # print("nb_of_clusters is", min_channel_num)
            if norm_type == "l1":
                weight_copy_sum = torch.sum(torch.reshape(model.module_list[idx][0].weight.data.abs(), (channels, -1)), dim=1)
            elif norm_type == "l2":
                weight_copy_sum = torch.norm(model.module_list[idx][0].weight.data.view(channels, -1), 2, 1)
            else:
                raise ValueError("norm type is illegal")
            # print("weight_copy_sum is", weight_copy_sum)
            mask = weight_copy_sum.gt(thre).float()
            if int(torch.sum(mask)) < min_channel_num:
                _, sorted_index_weights = torch.sort(weight_copy_sum, descending=True)
                mask[sorted_index_weights[:min_channel_num]] = 1.
            # print("at last mask is", mask)
            remain = int(mask.sum())
            pruned = pruned + mask.shape[0] - remain

            print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                  f'remaining channel: {remain:>4d}')
            # raise  ValueError("stop")
        else:
            mask = torch.ones(bn_module.weight.data.shape)
            remain = mask.shape[0]
            print(f'not prune layer info:  layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                  f'remaining channel: {remain:>4d}')
        total += mask.shape[0]
        num_filters.append(remain)
        filters_mask.append(mask.clone())
    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')
    return num_filters, filters_mask

def prune_soft_model(model, CBL_idx, CBLidx2mask):
    # model_copy = deepcopy(model)
    # model_copy = model
    for idx in CBL_idx:
        conv_model = model.module_list[idx][0]
        mask = CBLidx2mask[idx].cuda()
        weight_torch = conv_model.weight.data
        c, n, h, w = weight_torch.shape
        a = conv_model.weight.data.view(-1)
        kernel_length =n*h*w
        for id, x in enumerate(mask):
            a[id*kernel_length:(id+1)*kernel_length] *= x
        conv_model.weight.data =a.view(c, n, h ,w)

def prune_soft_model_code(model, CBL_idx, CBLidx2mask):
    for idx in CBL_idx:
        conv_model = model.module_list[idx][0]
        mask = CBLidx2mask[idx].cuda()
        weight_torch = conv_model.weight.data
        # print("before conv_model.weight.data is", conv_model.weight.data)
        if len(weight_torch.size()) == 4:
            c, n, h, w = weight_torch.shape
            weight_vec = conv_model.weight.data.view(-1)
            kernel_length =n*h*w
            for id, x in enumerate(mask):
                weight_vec[id*kernel_length:(id+1)*kernel_length] *= x
            conv_model.weight.data =weight_vec.view(c, n, h ,w)
        else:
            raise ValueError("this prune id {} is not conv type, please check prune id list ".format(idx))






