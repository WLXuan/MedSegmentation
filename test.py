import argparse
import copy
import logging
import os
import random
import sys
import time

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume, cvtColor, resize_image, preprocess_input, test_single_image
from models.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from models.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', default=True, action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit models')

parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f mean_iou %f' % (
            i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_iou = np.mean(metric_list, axis=0)[2]
    logging.info('Testing performance in best val models: mean_dice : %f mean_hd95 : %f mean_iou : %f' % (performance, mean_hd95, mean_iou))
    return "Testing Finished!"


def t(args, model):
    video_path = 'datasets/pred/cut0.mp4'
    video_save_path = 'datasets/pred/labeled5_cut0.mp4'
    video_fps = 30
    capture = cv2.VideoCapture(video_path)
    if video_save_path != "":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
    ref, frame = capture.read()
    if not ref:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

    fps = 0.0
    while (True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        if not ref:
            break
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        frame = np.array(detect_image(args, frame, model))
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        t2 = time.time()
        fps = 1 / np.round(t2 - t1, 3)
        # fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)
        # c = cv2.waitKey(1) & 0xff
        if video_save_path != "":
            out.write(frame)

        # if c == 27:
        #     capture.release()
        #     break
    print("Video Detection Done!")
    capture.release()
    if video_save_path != "":
        print("Save processed video to the path :" + video_save_path)
        out.release()
    cv2.destroyAllWindows()
    # image = Image.open(image_path)
    # image = image.convert('L')
    # image = np.array(image)
    # test_single_image(image, model, classes=args.num_classes,
    #                    patch_size=[args.img_size, args.img_size],
    #                    test_save_path='datasets', case='test0', z_spacing=args.z_spacing)


# ---------------------------------------------------#
#   检测图片
# ---------------------------------------------------#
def detect_image(args, image, model, count=False, name_classes=None):
    # ---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    # ---------------------------------------------------------#
    image = cvtColor(image)
    # ---------------------------------------------------#
    #   对输入图像进行一个备份，后面用于绘图
    # ---------------------------------------------------#
    old_img = copy.deepcopy(image)
    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]
    # ---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    # ---------------------------------------------------------#
    image_data, nw, nh = resize_image(image, (args.img_size, args.img_size))
    # ---------------------------------------------------------#
    #   添加上batch_size维度
    # ---------------------------------------------------------#
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        images = images.cuda()

        # ---------------------------------------------------#
        #   图片传入网络进行预测
        # ---------------------------------------------------#
        pr = model(images)[0]
        # ---------------------------------------------------#
        #   取出每一个像素点的种类
        # ---------------------------------------------------#
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        # --------------------------------------#
        #   将灰条部分截取掉
        # --------------------------------------#
        pr = pr[int((args.img_size - nh) // 2): int((args.img_size - nh) // 2 + nh), \
             int((args.img_size - nw) // 2): int((args.img_size - nw) // 2 + nw)]
        # ---------------------------------------------------#
        #   进行图片的resize
        # ---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        # ---------------------------------------------------#
        #   取出每一个像素点的种类
        # ---------------------------------------------------#
        pr = pr.argmax(axis=-1)

    # ---------------------------------------------------------#
    #   计数
    # ---------------------------------------------------------#
    if count:
        classes_nums = np.zeros([args.num_classes])
        total_points_num = orininal_h * orininal_w
        print('-' * 63)
        print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
        print('-' * 63)
        for i in range(args.num_classes):
            num = np.sum(pr == i)
            ratio = num / total_points_num * 100
            if num > 0:
                print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                print('-' * 63)
            classes_nums[i] = num
        print("classes_nums:", classes_nums)
    # if self.mix_type == 0:
    # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
    # for c in range(self.num_classes):
    #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
    #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
    #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
    colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
              (0, 128, 128),
              (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
              (192, 0, 128),
              (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
              (0, 64, 128),
              (128, 64, 12)]
    seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
    # ------------------------------------------------#
    #   将新图片转换成Image的形式
    # ------------------------------------------------#
    image = Image.fromarray(np.uint8(seg_img))
    # ------------------------------------------------#
    #   将新图与原图及进行混合
    # ------------------------------------------------#
    image = Image.blend(old_img, image, 0.7)

    # elif self.mix_type == 1:
    #     # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
    #     # for c in range(self.num_classes):
    #     #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
    #     #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
    #     #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
    #     seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
    #     # ------------------------------------------------#
    #     #   将新图片转换成Image的形式
    #     # ------------------------------------------------#
    #     image = Image.fromarray(np.uint8(seg_img))
    #
    # elif self.mix_type == 2:
    #     seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
    #     # ------------------------------------------------#
    #     #   将新图片转换成Image的形式
    #     # ------------------------------------------------#
    #     image = Image.fromarray(np.uint8(seg_img))

    return image


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': './data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "./models/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                              0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_' + str(args.max_epochs - 1))

    snapshot = './models/KaggleEndovis1TransU_Synapse224/AddTU_pretrain_R50-ViT-B_16_skip3_epo100_bs8_224/epoch_99.pth'
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/KEtest_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = './predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    # t(args, net)
    inference(args, net, test_save_path)
