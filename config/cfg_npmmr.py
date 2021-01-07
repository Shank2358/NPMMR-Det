# coding=utf-8

PROJECT_PATH = "./"
DATA_PATH = "/mnt/Datasets/DIOR/"

DATA = {"CLASSES":['airplane','airport','baseballfield','basketballcourt','bridge','chimney',
        'dam','Expressway-Service-area','Expressway-toll-station','golffield','groundtrackfield','harbor',
        'overpass','ship','stadium','storagetank','tenniscourt','trainstation','vehicle','windmill'],
        "NUM":20}

MODEL = {
        "ANCHORS":[[(1.494992296, 1.772419808), (2.550184278, 5.105188103), (4.511253175, 2.041398611)], # Anchors for small obj
                   [(3.852394468, 3.413543783), (3.827394513, 9.012606993), (7.569651633, 7.192874667)], # Anchors for medium obj
                   [(5.269568089, 8.068825014), (10.13079538, 3.44005408), (10.41848982, 10.60006263)]], # Anchors for big obj
        "STRIDES":[8, 16, 32],
        "ANCHORS_PER_SCLAE":3
        }

MAX_LABEL = 500
SHOW_HEATMAP = False
SCALE_FACTOR=2.0

TRAIN = {
         "EVAL_TYPE":'VOC', #['VOC', 'COCO']
         "TRAIN_IMG_SIZE":544,
         "TRAIN_IMG_NUM":5862,
         "AUGMENT":True,
         "MULTI_SCALE_TRAIN":True,
         "MULTI_TRAIN_RANGE":[10,20,1],
         "BATCH_SIZE":16,
         "IOU_THRESHOLD_LOSS":0.5,
         "EPOCHS":151,
         "NUMBER_WORKERS":16,
         "MOMENTUM":0.9,
         "WEIGHT_DECAY":0.0005,
         "LR_INIT":1.5e-4,
         "LR_END":1e-6,
         "WARMUP_EPOCHS":5,
         "IOU_TYPE":'CIOU' #['GIOU','CIOU']
         }

TEST = {
        "EVAL_TYPE":'COCO', #['VOC', 'COCO', 'BOTH']
        "EVAL_JSON":'test.json',
        "EVAL_NAME":'test',
        "NUM_VIS_IMG":100,
        "TEST_IMG_SIZE":544,
        "BATCH_SIZE":1,
        "NUMBER_WORKERS":16,
        "CONF_THRESH":0.05,
        "NMS_THRESH":0.5,
        "NMS_METHODS":'NMS_DIOU', #['NMS', 'SOFT_NMS', 'NMS_DIOU', #'NMS_DIOU_SCALE']
        "MULTI_SCALE_TEST":False,
        "MULTI_TEST_RANGE":[320,640,96],
        "FLIP_TEST":False
      }
