import argparse
import os
import cv2
import numpy as np
from glob import glob
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
from paddle.vision import transforms

from models.resnet import ResNet_MS3
from datasets.mvtec import MVTecDataset, load_gt
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection")
    # required training super-parameters
    parser.add_argument("--depth", type=int, default=18, help="student resnet depth")
    parser.add_argument("--checkpoint", type=str, default=None, help="student checkpoint")
    parser.add_argument("--category", type=str, default='leather', help="category name for MvTec AD dataset")
    parser.add_argument("--epochs", type=int, default=100, help='number of epochs')
    parser.add_argument("--lr", type=float, default=0.4, help='learning rate')
    parser.add_argument("--momentum", type=float, default=0.9, help='momentum')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight_decay')
    parser.add_argument("--checkpoint_epoch", type=int, default=100, help="checkpoint resumed for testing (1-based)")
    parser.add_argument("--batch_size", type=int, default=32, help='batch size')
    parser.add_argument("--train_val", type=bool, default=True, help='train with val')
    parser.add_argument("--print_freq", type=int, default=2)
    parser.add_argument("--compute_pro", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)

    # trivial parameters
    parser.add_argument("--result_path", type=str, default='results', help="save results")
    parser.add_argument("--save_fig", action='store_true', help="save images with anomaly score")
    parser.add_argument("--mvtec_ad", type=str, default='./data', help="MvTec-AD dataset path")
    parser.add_argument('--model_save_path', type=str, default='snapshots', help='path where student models are saved')

    args = parser.parse_args()

    np.random.seed(args.seed)
    paddle.seed(args.seed)

    # build model
    teacher = ResNet_MS3(depth=args.depth, pretrained=True)
    student = ResNet_MS3(depth=args.depth, pretrained=False)

    # build datasets
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_neg_image_list = sorted(glob(os.path.join(args.mvtec_ad, args.category, 'test', 'good', '*.png')))
    test_pos_image_list = set(glob(os.path.join(args.mvtec_ad, args.category, 'test', '*', '*.png'))) - set(
        test_neg_image_list)
    test_pos_image_list = sorted(list(test_pos_image_list))
    test_neg_dataset = MVTecDataset(test_neg_image_list, transform=transform)
    test_pos_dataset = MVTecDataset(test_pos_image_list, transform=transform)
    test_neg_loader = DataLoader(test_neg_dataset, batch_size=1, shuffle=False, drop_last=False)
    test_pos_loader = DataLoader(test_pos_dataset, batch_size=1, shuffle=False, drop_last=False)

    saved_dict = paddle.load(args.checkpoint)
    category = args.category
    gt = load_gt(args.mvtec_ad, category)

    print('load ' + args.checkpoint)
    student.load_dict(saved_dict)

    pos = test(teacher, student, test_pos_loader)
    neg = test(teacher, student, test_neg_loader)

    scores = []
    for i in range(len(pos)):
        temp = cv2.resize(pos[i], (256, 256))
        scores.append(temp)
    for i in range(len(neg)):
        temp = cv2.resize(neg[i], (256, 256))
        scores.append(temp)

    scores = np.stack(scores)
    neg_gt = np.zeros((len(neg), 256, 256), dtype=np.bool)
    gt_pixel = np.concatenate((gt, neg_gt), 0)
    gt_image = np.concatenate((np.ones(pos.shape[0], dtype=np.bool), np.zeros(neg.shape[0], dtype=np.bool)), 0)

    if args.compute_pro:
        pro = evaluate(gt_pixel, scores, metric='pro')
        auc_pixel = evaluate(gt_pixel.flatten(), scores.flatten(), metric='roc')
        auc_image_max = evaluate(gt_image, scores.max(-1).max(-1), metric='roc')
        print('Catergory: {:s}\tPixel-AUC: {:.6f}\tImage-AUC: {:.6f}\tPRO: {:.6f}'.format(category, auc_pixel,
                                                                                          auc_image_max, pro))
    else:
        auc_pixel = evaluate(gt_pixel.flatten(), scores.flatten(), metric='roc')
        auc_image_max = evaluate(gt_image, scores.max(-1).max(-1), metric='roc')
        print('Catergory: {:s}\tPixel-AUC: {:.6f}\tImage-AUC: {:.6f}'.format(category, auc_pixel, auc_image_max))


def test(teacher, student, loader):
    teacher.eval()
    student.eval()
    loss_map = np.zeros((len(loader.dataset), 64, 64))
    i = 0
    for batch_data in loader:
        _, batch_img = batch_data
        with paddle.no_grad():
            t_feat = teacher(batch_img)
            s_feat = student(batch_img)
        score_map = 1.
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], axis=1)
            s_feat[j] = F.normalize(s_feat[j], axis=1)
            sm = paddle.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
            sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)
            # aggregate score map by element-wise product
            score_map = score_map * sm
        loss_map[i: i + batch_img.shape[0]] = score_map.squeeze().cpu().numpy()
        i += batch_img.shape[0]
    return loss_map


if __name__ == "__main__":
    main()
