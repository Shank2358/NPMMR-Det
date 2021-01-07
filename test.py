import utils.gpu as gpu
from model.npmmrdet_model import NPMMRDet
from tensorboardX import SummaryWriter
from eval.evaluator import Evaluator
import argparse
import os
import config.cfg_npmmrdet_dior as cfg
from utils.visualize import *

import time
import logging
from utils.utils_coco import *
from utils.log import Logger
import cv2
from eval.coco_eval import COCOEvaluator

class Tester(object):
    def __init__(self, weight_path=None, gpu_id=0, visiual=None, eval=False):
        self.img_size = cfg.TEST["TEST_IMG_SIZE"]
        self.__num_class = cfg.DATA["NUM"]
        self.__conf_threshold = cfg.TEST["CONF_THRESH"]
        self.__nms_threshold = cfg.TEST["NMS_THRESH"]
        self.__device = gpu.select_device(gpu_id, force_cpu=False)
        self.__multi_scale_test = cfg.TEST["MULTI_SCALE_TEST"]
        self.__flip_test = cfg.TEST["FLIP_TEST"]
        self.__classes = cfg.DATA["CLASSES"]

        self.__visiual = visiual
        self.__eval = eval
        self.__model = NPMMRDet().to(self.__device)  # Single GPU

        net_model = NPMMRDet()
        if torch.cuda.device_count() >1: ## Multi GPUs
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net_model = torch.nn.DataParallel(net_model) ## Multi GPUs
            self.__model = net_model.to(self.__device)
        elif torch.cuda.device_count() ==1:
            self.__model = net_model.to(self.__device)

        self.__load_model_weights(weight_path)

        self.__evalter = Evaluator(self.__model, visiual=False)

    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))
        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model.load_state_dict(chkpt) #['model']
        #print("loading weight file is done")
        del chkpt

    def test(self):
        global logger
        logger.info("***********Start Evaluation****************")

        if self.__visiual:
            imgs = os.listdir(self.__visiual)
            for v in imgs:
                path = os.path.join(self.__visiual, v)
                #print("test images : {}".format(path))
                img = cv2.imread(path)
                assert img is not None
                bboxes_prd = self.__evalter.get_bbox(img)
                if bboxes_prd.shape[0] != 0:
                    boxes = bboxes_prd[..., :4]
                    class_inds = bboxes_prd[..., 5].astype(np.int32)
                    scores = bboxes_prd[..., 4]
                    visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.__classes)
                    path = os.path.join(cfg.PROJECT_PATH, "prediction/imgs_all/{}".format(v))
                    cv2.imwrite(path, img)
                    #print("saved images : {}".format(path))

        mAP = 0
        if self.__eval and cfg.TEST["EVAL_TYPE"] == 'VOC':
            with torch.no_grad():
                start = time.time()
                APs, inference_time = Evaluator(self.__model).APs_voc(self.__multi_scale_test, self.__flip_test)

                for i in APs:
                    print("{} --> AP : {}".format(i, APs[i]))
                    mAP += APs[i]
                mAP = mAP / self.__num_class
                logger.info('mAP:{}'.format(mAP))
                logger.info("inference time: {:.2f} ms".format(inference_time))
                writer.add_scalar('test/VOCmAP', mAP)
                end = time.time()
                logger.info("Test cost time:{:.4f}s".format(end - start))
                #print('mAP:%g' % (mAP))
                #print("inference time : {:.2f} ms".format(inference_time))

        elif self.__eval and cfg.TEST["EVAL_TYPE"] == 'COCO':
            with torch.no_grad():
                start = time.time()
                evaluator = COCOEvaluator(data_dir=cfg.DATA_PATH,
                                          img_size=cfg.TEST["TEST_IMG_SIZE"],
                                          confthre=cfg.TEST["CONF_THRESH"],
                                          nmsthre=cfg.TEST["NMS_THRESH"])
                ap50_95, ap50, inference_time = evaluator.evaluate(self.__model)
                logger.info('ap50_95:{} | ap50:{}'.format(ap50_95, ap50))
                logger.info("inference time: {:.2f} ms".format(inference_time))
                writer.add_scalar('test/COCOAP50', ap50)
                writer.add_scalar('test/COCOAP50_95', ap50_95)
                end = time.time()
                logger.info("Test cost time:{:.4f}s".format(end - start))

        elif self.__eval and cfg.TEST["EVAL_TYPE"] == 'BOTH':
            with torch.no_grad():
                start = time.time()
                APs, inference_time = Evaluator(self.__model).APs_voc(self.__multi_scale_test, self.__flip_test)
                for i in APs:
                    print("{} --> mAP : {}".format(i, APs[i]))
                    mAP += APs[i]
                mAP = mAP / self.__num_class
                logger.info('mAP:{}'.format(mAP))
                logger.info("inference time: {:.2f} ms".format(inference_time))
                writer.add_scalar('test/VOCmAP', mAP)
                end = time.time()
                logger.info("Test cost time:{:.4f}s".format(end - start))
                start = time.time()
                evaluator = COCOEvaluator(data_dir=cfg.DATA_PATH,
                                          img_size=cfg.TEST["TEST_IMG_SIZE"],
                                          confthre=cfg.TEST["CONF_THRESH"],
                                          nmsthre=cfg.TEST["NMS_THRESH"])
                ap50_95, ap50, inference_time = evaluator.evaluate(self.__model)
                logger.info('ap50_95:{} | ap50:{}'.format(ap50_95, ap50))
                logger.info("inference time: {:.2f} ms".format(inference_time))
                writer.add_scalar('test/COCOAP50', ap50)
                writer.add_scalar('test/COCOAP50_95', ap50_95)
                end = time.time()
                logger.info("Test cost time:{:.4f}s".format(end - start))

if __name__ == "__main__":
    global logger
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/best.pt', help='weight file path')
    parser.add_argument('--log_val_path', type=str, default='log/', help='weight file path')
    parser.add_argument('--visiual', type=str, default=None, help='test data path or None')
                        #default=''I:/Datasets/Detection/DIOR/JPEGImages''
    parser.add_argument('--eval', action='store_true', default=True, help='eval flag')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--log_path', type=str, default='log/', help='log path')
    opt = parser.parse_args()
    writer = SummaryWriter(logdir=opt.log_path + '/event')
    logger = Logger(log_file_name=opt.log_val_path + '/log_coco_test.txt', log_level=logging.DEBUG,
                    logger_name='NPMMRDet').get_log()

    Tester(weight_path=opt.weight_path, gpu_id=opt.gpu_id, eval=opt.eval, visiual=opt.visiual).test()