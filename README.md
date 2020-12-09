20201128 updates:

    I want to duplicate the yolo algorithm based on my own ideas, and structed prune yolo use pytorch, if you agree with my work, thank u to give me a star to support


20201209 updates:

    add prune code ,finetune code,  knowledge distill code.240Mb

    paper: Learning Efficient Convolutional Networks through Network Slimming

    reference code: https://github.com/tanluren/yolov3-channel-and-layer-pruning  (in this project, yolov4 voc dataset map only 0.83, but my project best map is 0.87 )

    how to prune: see ths last in readme.md

    next stage: i will use dynamic prune yolov4 and commit code

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

how to prune:

    1. use my code train a best ap model

    2. set best model as pretrained_weights in train_prune.py,                                then:   python train_prune.py    -> get a sparse model

    3. after second stage, u will get a sparse model, set this model in slim_prune.py         then:   python slim_prune.py     -> get a structed prune model

    4. u will obtain new yolov4.cfg and yolov4.weights, finetune model                        then:   python train.py