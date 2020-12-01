import torch.nn.functional as F

# from utils.google_utils import *
from utils.parse_config import *
from utils.util import *


class FeatureConcat(nn.Module):
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


def create_modules(module_defs, img_size, arc, algorithm):
    # Constructs module list of layer blocks from module configuration in module_defs

    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layes
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            kernel_size = int(mdef['size'])
            pad = (kernel_size - 1) // 2 if int(mdef['pad']) else 0
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=kernel_size,
                                                   stride=int(mdef['stride']),
                                                   padding=pad,
                                                   bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'leaky':  # TODO: activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                # modules.add_module('activation', Swish())
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())

        elif mdef['type'] == 'maxpool':
            kernel_size = int(mdef['size'])
            stride = int(mdef['stride'])
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            if kernel_size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=int(mdef['stride']), mode='nearest')

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in mdef['layers'].split(',')]
            # print("route layers model def is", mdef['layers'])
            # print("route layers is", layers)
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            if 'groups' in mdef:
                filters = filters // 2
            routs.extend([l if l > 0 else l + i for l in layers])
            # if mdef[i+1]['type'] == 'reorg3d':
            #     modules = nn.Upsample(scale_factor=1/float(mdef[i+1]['stride']), mode='nearest')  # reorg3d

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            filters = output_filters[int(mdef['from'])]
            layer = int(mdef['from'])
            routs.extend([i + layer if layer < 0 else layer])

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            stride = [8, 16, 32]  # P5, P4, P3 strides
            if algorithm == 'v3':
                stride = [32, 16, 8]  # P5, P4, P3 strides yolov3
            elif algorithm == "v4":
                stride = [8, 16, 32]  # P5, P4, P3 strides
            elif algorithm == "v5":
                stride = [8, 16, 32]  # P5, P4, P3 strides
            else:
                raise ValueError("algorithm only support yolov3 v4 v5")
            mask = [int(x) for x in mdef['mask'].split(',')]  # anchor mask
            # print("anchors is", mdef['anchors'])
            # print("mask is ", mdef['mask'])
            modules = YOLOLayer(anchors=mdef['anchors'][mask],# anchor list
                                num_classes=int(mdef['classes']),  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1 or 2
                                stride=stride[yolo_index]
                                )  # yolo architecture
            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                if arc == 'defaultpw' or arc == 'Fdefaultpw':  # default with positive weights
                    b = [-4, -3.6]  # obj, cls
                elif arc == 'default':  # default no pw (40 cls, 80 obj)
                    b = [-5.5, -4.0]
                elif arc == 'uBCE':  # unified BCE (80 classes)
                    b = [0, -8.5]
                elif arc == 'uCE':  # unified CE (1 background + 80 classes)
                    b = [10, -0.1]
                elif arc == 'Fdefault':  # Focal default no pw (28 cls, 21 obj, no pw)
                    b = [-2.1, -1.8]
                elif arc == 'uFBCE' or arc == 'uFBCEpw':  # unified FocalBCE (5120 obj, 80 classes)
                    b = [0, -6.5]
                elif arc == 'uFCE':  # unified FocalCE (64 cls, 1 background + 80 classes)
                    b = [7.7, -1.1]

                bias = module_list[-1][0].bias.view(len(mask), -1)  # 255 to 3x85
                bias[:, 4] += b[0] - bias[:, 4].mean()  # obj
                bias[:, 5:] += b[1] - bias[:, 5:].mean()  # cls
                # bias = torch.load('weights/yolov3-spp.bias.pt')[yolo_index]  # list of tensors [3x85, 3x85, 3x85]
                module_list[-1][0].bias = torch.nn.Parameter(bias.view(-1))
                # utils.print_model_biases(model)
            except:
                print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list, routs, hyperparams


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x.mul(torch.tanh(F.softplus(x)))


class YOLOLayer(nn.Module):
    def __init__(self, anchors=[], num_classes=80, img_size=608, stride=32, yolo_index=0, model_out=False):
        super(YOLOLayer, self).__init__()

        self.num_classes = num_classes
        self.anchors = torch.Tensor(anchors).cuda()
        self.num_anchors = len(anchors)
        self.stride = stride
        self.img_size = img_size
        self.anchor_vec = self.anchors/self.stride
        self.anchor_wh = self.anchor_vec.view(1, 3, 1, 1, 2)
        # self.anchor_wh = self.anchor_vec.view(1, self.num_anchors, 1, 1, 2)
        self.nb_x_grid = 0
        self.nb_y_grid = 0
        self.nb_grid = torch.tensor((img_size/stride, img_size/stride),dtype=torch.float).cuda()


    def create_grids(self, nb_grids=(13, 13), device='cpu'):
        self.nb_x_grid, self.nb_y_grid = nb_grids
        # self.nb_grid = torch.tensor(nb_grids, dtype=torch.float)
        # print("self.nb_grid is", self.nb_grid )
        # if not self.tarining:
        yv, xv = torch.meshgrid([torch.arange(self.nb_y_grid, device=device), torch.arange(self.nb_x_grid, device=device)])
        self.grid = torch.stack((xv, yv), dim=2).view((1, 1, self.nb_y_grid, self.nb_x_grid, 2))

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, pred):
        # print("pred shape is", pred.shape)
        batch_size, grid_y, grid_x = pred.shape[0], pred.shape[-2], pred.shape[-1]
        if (self.nb_x_grid, self.nb_y_grid) != (grid_x, grid_y):
            self.create_grids((grid_x, grid_y), pred.device)
        pred = pred.view(batch_size, self.num_anchors, (self.num_classes+5), self.nb_y_grid, self.nb_x_grid)
        # print("pred shape is", pred.shape)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous() # change [batch_size, num_anchors, class_num+5, grid_x, grid_y] ->  [batch_size, num_anchors, grid_x, grid_y, class_num+5]
        if self.training:
            return pred
        else:# return inference pred and training pred
            pred_result = pred.clone()   # shape: [batch_size, num_anchors, grid_x, grid_szie, class_num+5]
            # print("self.grid is", self.grid)
            # print("self.grid is", self.grid.shape)
            # print("torch.sigmoid(pred_result[..., :2]) is",torch.sigmoid(pred_result[..., :2]).shape )
            # print("torch.sigmoid(pred_result[..., :2])  is",torch.sigmoid(pred_result[..., :2]) )
            ## yolov3 yolo layer
            pred_result[..., :2] = torch.sigmoid(pred_result[..., :2]) + self.grid    #  [batch_size, num_anchors, grid_x, grid_szie, 2(x,y)] + [grid_x, grid_y, 2(grid_x_loc + grid_y_loc)]
            # pred_result[..., 2:4] = torch.exp(pred_result[..., 2:4]) * self.anchor_wh #  exp([batch_size, num_anchors, grid_x, grid_szie, 2(w,h)]) * ori_w_h/stride.reshape->[1, num_anchors, grid_x, grid_y, 2(anchor_w_h)]
            pred_result[..., 2:4] = torch.exp(pred_result[..., 2:4]) * self.anchor_wh #  exp([batch_size, num_anchors, grid_x, grid_szie, 2(w,h)]) * ori_w_h/stride.reshape->[1, num_anchors, grid_x, grid_y, 2(anchor_w_h)]
            pred_result[..., :4] *= self.stride                                       # feature_map x,y,w,h -> ori x,y,w,h
            torch.sigmoid_(pred_result[..., 4:])                                      # sigmod(conf_and_class_prob)
            ## yolov4 or yolov5 yolo layer
            # y = pred_result.sigmoid()
            # y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid.to(pred_result.device)) * self.stride  # xy
            # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_wh  # wh

            return pred_result.view(batch_size, -1, self.num_classes+5)  , pred             # [batch_size, grid_size*grid_size*3, num_classes+5]   reshape from [1, 3, 13, 13, 85] to [1, 507, 85]


class Darknet(nn.Module):
    # YOLO object detection model

    def __init__(self, cfg, img_size=(608, 608), algorithm='v3', arc='default'):
        super(Darknet, self).__init__()
        if isinstance(cfg, str):
            self.module_defs = parse_model_cfg(cfg)
            # print(" self.module_defs  is ", self.module_defs )
        elif isinstance(cfg, list):
            self.module_defs = cfg   #  type: list-dict #[{'type': 'convolutional', 'batch_normalize': '1', 'filters': '32', 'size': '3', 'stride': '1', 'pad': '1', 'activation': 'leaky'},
        self.module_list, self.routs, self.hyperparams = create_modules(self.module_defs, img_size, arc, algorithm)  # layer lisrt
        self.yolo_layers = get_yolo_layers(self)
        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

    def forward(self, x, var=None):
        img_size = x.shape[-2:]
        layer_outputs = []
        output = []
        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                    if 'groups' in mdef:
                        x = x[:, (x.shape[1] // 2):]
                else:
                    try:
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    # print(''), [print(layer_outputs[i].shape) for i in layers], print(x.shape)
            elif mtype == 'shortcut':
                x = x + layer_outputs[int(mdef['from'])]
            elif mtype == 'yolo':
                # x = module(x, img_size)
                x = module(x)
                output.append(x)
            layer_outputs.append(x if i in self.routs else [])

        if self.training:
            return output
        else:
            inference_out, training_out = list(zip(*output))  # inference output, training output
            return torch.cat(inference_out, 1), training_out

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        # model_info(self)  # yolov3-spp reduced from 225 to 152 layers


def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for yolov3   [139, 150, 161] for yolov4


def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw

def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        save_weights(model, path='converted.weights', cutoff=-1)
        print("Success: converted '%s' to 'converted.weights'" % weights)

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        torch.save(chkpt, 'converted.pt')
        print("Success: converted '%s' to 'converted.pt'" % weights)

    else:
        print('Error: extension not supported.')


if __name__ == "__main__":
    convert(cfg='cfg/yolov3.cfg', weights='weights/yolov3.weights')

