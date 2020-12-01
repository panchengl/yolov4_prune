import torch
import torch.nn as nn
import math

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou



class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'None'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true)*(1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    # print("wh1 - anchor is", wh1)
    # print("wh2 - gt_wh_grid is", wh2)
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    # print("wh1 iterator - anchor is", wh1)
    # print("wh2 iterator - gt_wh_grid is", wh2)
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    # print("inter is ", inter)
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)

def build_targets_error(preds, targets, model):
    # print("targets shape is", targets.shape)
    number_gt = targets.shape[0]   #  number_gt_box
    target_cls, target_box, indices, anchor_ve =[],  [], [], []
    use_all_anchor,reject = True, True
    gain = torch.ones(6, device=targets.device)
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for i, j in enumerate(model.module.yolo_layers):  # use w h
        anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec

        gain[2:] = torch.tensor(preds[i].shape)[[3,2,3,2]]  # nomilizar location -> grid size scale (xmin:0.002 grid: 40, scale40)
        target_in_grid, a = targets * gain, []   # normilizar location * grid_size -> grid size gt (true xmin=40*0.002=0.08) shape[N, 6]
        target_wh_in_grid = target_in_grid[:, 4:6]
        if number_gt:
            iou_matrix_gt_anchors = wh_iou(anchors, target_wh_in_grid)  # anchors:(3, 2) gwh:(n,2) -> iou(3,n)  every gt_label clac 3 anchors iou,
            # print("this yolo layer iou matrix is", iou_matrix_gt_anchors)
            #
            if use_all_anchor:
                anchors_num = anchors.shape[0]
                a = torch.arange(anchors_num).view(-1, 1).repeat(1, number_gt).view(-1)
                # print("ori target_in_grid is", target_in_grid)   # shape(num_gt, 6)     [[batch_id, label, xcenter_id, y_center_grid_id], [,...]]
                target_in_grid = target_in_grid.repeat(anchors_num, 1)    #  shape(num_gt*anchors_num, 6),   the first axis expand anchors_num bei and second axis don`t change
                # print("target_in_grid is", target_in_grid)
            else:
                iou_matrix_gt_anchors, a = iou_matrix_gt_anchors.max(0)

            if reject:
                j = iou_matrix_gt_anchors.view(-1) > model.module.hyp['iou_t'] # every gt with every anchors iou > thr : 3anchors 3gt
                                                                               # [False, False, True, False, False, False]: only No1 gt with No3 anchors iou>5
                targets_selected = target_in_grid[j]                           # becareful, after repeat, target_in_grid shape is [N*_anchors_num, 6]
                a = a[j]
                # print("j is", j)
        # print("target_selected is", targets_selected)
        batch_img_id, batch_img_labels = targets_selected[:, :2].long().t()    # b: batch img id,  c:batch img `s  targets label  after transpose: [1, N(selected)]
        # print("b is", batch_img_id)
        # print("c is", batch_img_labels)
        selected_xy = targets_selected[:, 2:4]
        gt_wh = targets_selected[:, 4:6]
        grid_x, grid_y = selected_xy.long().t()   #grid x, y indices
        #  batch_img_id: img_id in this batch
        #  a: every gt with anchor > iou_thr anchor id
        #  grid_x, grid_y: ...
        indices.append((batch_img_id, a, grid_x, grid_y))

        gt_xy = selected_xy -  selected_xy.floor()
        target_box.append(torch.cat((gt_xy, gt_wh), dim=1))   # gt is:  grid_x, grid_y [x_center_ingrid-grid_x, y_center_grid-grid_y, w_in_grid, h_in_grid]
        anchor_ve.append(anchors[a])

        target_cls.append(batch_img_labels)
        if batch_img_labels.shape[0]:
            assert batch_img_labels.max() < model.module.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           model.module.nc, model.module.nc - 1, batch_img_labels.max())
    return  target_cls, target_box, indices, anchor_ve

def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    nt = targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets

    style = None
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    for i, j in enumerate(model.yolo_layers):

        anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        na = anchors.shape[0]  # number of anchors
        at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)

        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        if nt:
            # r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
            # j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t'] # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            # j = wh_iou(anchors, t[:, 4:6]) > model.module.hyp['iou_t'] if multi_gpu else model.hyp['iou_t']# iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

            # overlaps
            gxy = t[:, 2:4]  # grid xy
            z = torch.zeros_like(gxy)
            if style == 'rect2':
                g = 0.2  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                a, t = torch.cat((a, a[j], a[k]), 0), torch.cat((t, t[j], t[k]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1]), 0) * g

            elif style == 'rect4':
                g = 0.5  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class
        if c.shape[0]:  # if any targets
            assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           model.nc, model.nc - 1, c.max())

    return tcls, tbox, indices, anch

# def build_targets(p, targets, model):
#     # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
#     # det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
#     na, nt = det.na, targets.shape[0]  # number of anchors, targets
#     tcls, tbox, indices, anch = [], [], [], []
#     gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
#     ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
#     targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
#
#     g = 0.5  # bias
#     off = torch.tensor([[0, 0],
#                         [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
#                         # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
#                         ], device=targets.device).float() * g  # offsets
#
#     # for i in range(det.nl):
#     for i, j in enumerate(model.yolo_layers):
#         anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
#         gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
#
#         # Match targets to anchors
#         t = targets * gain
#         if nt:
#             # Matches
#             r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
#             j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
#             # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
#             t = t[j]  # filter
#
#             # Offsets
#             gxy = t[:, 2:4]  # grid xy
#             gxi = gain[[2, 3]] - gxy  # inverse
#             j, k = ((gxy % 1. < g) & (gxy > 1.)).T
#             l, m = ((gxi % 1. < g) & (gxi > 1.)).T
#             j = torch.stack((torch.ones_like(j), j, k, l, m))
#             t = t.repeat((5, 1, 1))[j]
#             offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
#         else:
#             t = targets[0]
#             offsets = 0
#
#         # Define
#         b, c = t[:, :2].long().T  # image, class
#         gxy = t[:, 2:4]  # grid xy
#         gwh = t[:, 4:6]  # grid wh
#         gij = (gxy - offsets).long()
#         gi, gj = gij.T  # grid xy indices
#
#         # Append
#         a = t[:, 6].long()  # anchor indices
#         indices.append((b, a, gj.clamp_(0, gain[3]), gi.clamp_(0, gain[2])))  # image, anchor, grid indices
#         tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#         anch.append(anchors[a])  # anchors
#         tcls.append(c)  # class
#
#     return tcls, tbox, indices, anch

def compute_loss(preds, targets, model):
    # targets = [image, class, x, y, w, h]
    # print("model yolo layer is", model.module.yolo_layers)
    ft = torch.cuda.FloatTensor if preds[0].is_cuda else torch.Tensor
    from utils.torch_utils import is_parallel
    # h = model.module.hyp if is_parallel(model) else model.hyp
    h = model.hyp
    # tcls, tbox, indices , anchor_vec = build_targets(preds, targets, model)

    loss_cls, loss_box, loss_obj = ft([0]), ft([0]), ft([0])
    target_cls, target_box, indices, anchor_vec = build_targets(preds, targets, model)
    # target_cls, target_box, indices, anchor_vec = build_targets_v3_channels(model, targets)
    red = "mean"
    BCE_cls = nn.BCEWithLogitsLoss(pos_weight=ft([ h['cls_pw'] ]), reduction="mean")
    BCE_obj = nn.BCEWithLogitsLoss(pos_weight=ft([ h['obj_pw'] ]), reduction="mean")


    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)


    gamma = h['fl_gamma']
    if gamma:
        BCE_cls, BCE_obj = FocalLoss(BCE_cls, gamma), FocalLoss(BCE_obj, gamma)

    nb_grid_points , nb_targets = 0, 0
    for i, pi in enumerate(preds):
        batch_img_id, anchor_id, grid_x, grid_y = indices[i] # img, anchor, gridx, gridy
        target_obj = torch.zeros_like(pi[..., 0])  # create [batch_size, num_anchors, grid_x, grid_y] all obj = 0
        # print("this layer pi is", pi)
        # print("this layer pi shape is", pi.shape)
        # print("target_obj shape is", target_obj.shape)
        # print("target_obj  is", target_obj)
        nb_grid_points += target_obj.numel()

        nb = len(batch_img_id)   # numbers of gt with anchors iou>thr
        if nb :
            nb_targets += nb
            # print("pi shape is", pi.shape)
            # print("batch_img_id is", batch_img_id)
            # print("anchor_id is", anchor_id)
            # print("grid_x is", grid_x)
            ps = pi[batch_img_id, anchor_id, grid_x, grid_y]    # pi shape [batch_size, num_anchors, grid_x, grid_y, class_num+5]  ps shape: [len(batch_img_id), num_class+5] asser len(batch_img_id)==len(ancho_vec)==len(grid_x)

            # #Giou yolov3 calc
            pred_xy = torch.sigmoid(ps[:, 0:2])
            pred_wh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchor_vec[i]  # anchor_vec:scale in ori_img; feature_map_wh*scale->ori_wh

            # Giou yolov4 v5  calc
            # pred_xy = ps[:, :2].sigmoid() * 2. - 0.5
            # pred_wh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchor_vec[i]

            pred_box = torch.cat((pred_xy, pred_wh), dim=1)
            #calc anchors with target iou>thr location  -> pred and  targets iou loss
            giou = bbox_iou(pred_box.t(), target_box[i], x1y1x2y2=False, CIoU=True) #
            loss_box += (1.0 - giou).sum() if red == 'sum' else (1.0 - giou).mean()
            # calc  anchors with gt iou>iou_thr location  giou loss
            target_obj[batch_img_id, anchor_id, grid_x, grid_y] = (1.0- model.gr) + model.gr * giou.detach().clamp(0).type(target_obj.dtype)

            if model.nc > 1:
                # print("ps shape is", ps.shape)
                t = torch.full_like(ps[:, 5:], cn)  # label smoth: target class prob set smooth value, init=0
                                                    # shape: [batch_size, num_anchors, grid_x, grid_y, 5]
                # print("ori t shape is", t.shape)
                # print("target_cls[i] is", target_cls[i])
                t[range(nb), target_cls[i]] = cp    # class prob location set 1
                loss_cls += BCE_cls(ps[:, 5:], t)

        loss_obj += BCE_obj(pi[..., 4], target_obj)  # obj use giou or iou as prob gt

    loss_box *= h['giou']
    loss_obj *= h['obj']
    loss_cls *= h['cls']
    if red == 'sum':
        bs = target_obj.shape[0]  # batch size
        loss_obj *= 3 / (6300 * bs) * 2  # 3 / np * 2
        if nb_targets:
            loss_cls *= 3 / nb_targets / model.nc
            loss_obj *= 3 / nb_targets
    loss = loss_box + loss_obj + loss_cls
    return loss , torch.cat((loss_box, loss_obj, loss_cls, loss)).detach()


