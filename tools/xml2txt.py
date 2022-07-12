import os
import argparse
import xml.etree.ElementTree as ET

def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):

    #更改为你自己的类别
    classes = ['airplane','airport','baseballfield','basketballcourt','bridge','chimney',
        'dam','Expressway-Service-area','Expressway-toll-station','golffield','groundtrackfield','harbor',
        'overpass','ship','stadium','storagetank','tenniscourt','trainstation','vehicle','windmill']
    img_inds_file = os.path.join(data_path, 'ImageSets', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]
    aaa=0
    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and (int(difficult) == 2):#and(int(difficult) == 1)
                    continue
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.strip())#.lower()
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                if xmax != xmin and ymax != ymin:
                    ann = image_ind
                    annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            print(annotation)
            if annotation != image_path:
                f.write(annotation + "\n")
    print(aaa)
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/mnt/Datasets/DIOR/") ##the path of DIOR dataset: /mnt/Datasets/DIOR/
    '''
        DATA_PATH = "/mnt/Datasets/DIOR/"
        ├── ...
        ├── JPEGImages
        |   ├── 000001.jpg
        |   ├── 000002.jpg
        |   └── ...
        ├── Annotations
        |   ├── 000001.xml
        |   ├── 000002.xml
        |   └── ...
        ├── ImageSets
            ├── test.txt (testing filename)
                ├── 000001
                ├── 000002
                └── ...
    '''
    #parser.add_argument("--train_annotation", default="/mnt/Datasets/DIOR/train.txt")##the output path of train set
    flags = parser.parse_args()
    if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
    num = convert_voc_annotation(os.path.join(flags.data_path), 'train', flags.train_annotation, False) #convert xml to txt
