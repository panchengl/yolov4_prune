import xml.etree.ElementTree as ET
import tqdm
import os

new_train_dir = ['/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/河北电力图片20191025-ok', '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/河北电力图片20191126-164-ok',
                 '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/华东电力平台20200512-ok', '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/华东电力平台20200609-ok',
                 '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/华东电力平台20200619-ok', '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/华云-烟火20200319-ok',
                 '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/龙瑞绍兴电力图片20191126-102-ok', '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/南网平台20200429-ok/有目标',
                 '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/山东测试20190121', '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/山东测试20191230-ok',
                 '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/深圳电力20181220', '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/施工车辆-烟雾20200403 -oyq-729-ok',
                 '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/温州电力平台20200828', '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/温州平台20200807',
                 '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/宿迁电力图片20191126-101-ok', '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/宿迁电力项目图片20191025-ok',
                 '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/烟火照片20200305-ok', '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/烟雾200327-ok',
                 '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/云南_正常报警_漏报_误报20200605', '/home/pcl/new_data/yolo_datasets/images/04.电力/未标金具类型/烟火20201012-ok']
test_img_dir = ["/home/pcl/pytorch_work/yolov5/ori_dianli/test/images/"]

train_txt = "data/train2017_txt_dataset.txt"
test_txt = "data/val2017_txt_dataset.txt"

orignal_class = ["DiaoChe", "TaDiao", "TuiTuJi", "BengChe", "WaJueJi", "ChanChe", "ShanHuo", "YanWu", "SuLiaoBu", "ShiGongJiXie", "FengZheng"]
class_sgjx = ["TuiTuJi", "BengChe", "WaJueJi", "ChanChe", "ShiGongJiXie"]
class_yanhuo = ["YanWu", "ShanHuo"]
class_suliaobu = ["SuLiaoBu", "FengZheng"]
class_zhongkeyuan = ["DiaoChe", "TaDiao", "ShiGongJiXie", "YanHuo", "SuLiaoBu"]

img_list_train = []
img_list_test = []

def get_files_in_all_dirs(all_dirs, img_list):
    for simple_dir in all_dirs:
        for root, dirs, files in os.walk(simple_dir):
            if len(files) != 0:
                if "ALL" not in root:
                    print("current deal dirs is", root)
                    for i in files:
                        file_type = i.split('.')[-1]
                        if file_type == 'JPG' or file_type == 'jpeg' or file_type == 'jpg':
                            try:
                                f = open(   os.path.join(root, i).replace(file_type, "xml") )
                                img_list.append(os.path.join(root, i) )
                            except:
                                print("this xml cannot open", os.path.join(root, i).replace(file_type, "xml"))
                                # raise ValueError("cannot open file")
                                continue
    return img_list

def convert_annotation(image_id, list_file):
    anno_file = image_id.replace('jpg', 'xml').replace('JPG', 'xml').replace('jpeg', 'xml')
    # print("annof_ile is", anno_file)
    try:
        in_file = open(anno_file)
    except:
        print(anno_file)
        print('error')
        raise  ValueError("error")
        return  0
    tree=ET.parse(in_file)
    root = tree.getroot()
    for size in root.iter('size'):
        width = int(size.find('width').text)
        height = int(size.find('height').text)
    list_file.write(" " + str(width) + " " + str(height))
    for obj in root.iter('object'):
        try:
            difficult = obj.find('difficult').text
        except:
            difficult = 0
        cls = obj.find('name').text
        if cls not in orignal_class:
            continue
        try:
            if cls in class_yanhuo:
                # print("orignal class is %s but change YanHuo" % cls)
                cls = "YanHuo"
            if cls in class_sgjx:
                # print("orignal class is %s but change ShiGongJiXie" % cls)
                cls = "ShiGongJiXie"
            if cls in class_suliaobu:
                # print("orignal class is %s but change SuLiaoBu" % cls)
                cls = "SuLiaoBu"
            cls_id = class_zhongkeyuan.index(cls)
        except:
            continue
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + str(cls_id) + " " + " ".join([str(a) for a in b]))



def create_dataset(file_list, txt_name):
    list_file = open(txt_name, 'w')
    num = 0
    for image_id in file_list:
        print("current id is",image_id)
        num = num + 1
        if(num %1000 == 0):
            print("current deal img_num is %d"%num)
        list_file.write(str(num) + ' ' + '%s' %(image_id) )
        a = convert_annotation(image_id, list_file)
        # num = num + 1
        list_file.write('\n')
    list_file.close()
    print("last deal img_num is", num)


if __name__ == "__main__":
    img_list_train = get_files_in_all_dirs(new_train_dir, img_list_train)
    img_list_test = get_files_in_all_dirs(test_img_dir, img_list_test)
    create_dataset(img_list_train, train_txt)
    create_dataset(img_list_test, test_txt)