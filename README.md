20201128 updates:

    I want to duplicate the yolo algorithm based on my own ideas, and structed prune yolo use pytorch, if you agree with my work, thank u to give me a star to support


voc datasets:

                        map

            yolov4      0.87(after 3 epoch)

            yolov3      0.83(after 10 epoch)

        u can easily obtain this performance


how to train:

    train command:

        python -m torch.distributed.launch --nproc_per_node 4 train_mutil_gpu.py --cfg cfg/yolov4.cfg --weights weights/yolov4.weights --device 0,1,2,3 --train_path data/train.txt --val_path data/val.txt --names_classes data/coco.names


# yolov4_prune
