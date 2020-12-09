20201128 updates:

    I want to duplicate the yolo algorithm based on my own ideas, and structed prune yolo use pytorch, if you agree with my work, thank u to give me a star to support


voc datasets:

                        map                        size

            yolov4      0.87(after 3 epoch)        240Mb

            yolov3      0.83(after 10 epoch)       240Mb

      yolov4_prune      0.832                       75Mb

        u can easily obtain this performance



how to train your own dataset:

1. create dataset, make your dataset become txt file- just like this:

    img_id img_dir width height label xmin ymin xmax ymax

    1 dir/img1.jpg width height label1 xmin ymin xmax ymax label2 xmin ymin xmax ymax .......

    2 dir/img2.jpg width height label1 xmin ymin xmax ymax

2.create label.names just like data/voc.names in my code


3.train command:

        python train.py --cfg cfg/yolov4.cfg --weights weights/yolov4.weights --device 0,1,2,3 --train_path data/train.txt --val_path data/val.txt --names_classes data/coco.names


# yolov4_prune
