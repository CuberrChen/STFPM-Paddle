import argparse
import os
import cv2
import time
import datetime
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

import paddle
import paddle.nn.functional as F
import paddle.optimizer as optim
from paddle.io import DataLoader
from paddle.vision import transforms

from models.resnet import ResNet_MS3
from datasets.mvtec import MVTecDataset, load_gt
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection")
    parser.add_argument("split", nargs="?", choices=["train", "test"])
    # required training super-parameters
    parser.add_argument("--depth", type=int, default=18, help="student resnet depth")
    parser.add_argument("--checkpoint", type=str, default=None, help="student checkpoint")
    parser.add_argument("--category", type=str , default='leather', help="category name for MvTec AD dataset")
    parser.add_argument("--epochs", type=int, default=100, help='number of epochs')
    parser.add_argument("--lr", type=float, default=0.4, help='learning rate')
    parser.add_argument("--momentum", type=float, default=0.9, help='momentum')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight_decay')
    parser.add_argument("--checkpoint_epoch", type=int, default=100, help="checkpoint resumed for testing (1-based)")
    parser.add_argument("--batch_size", type=int, default=32, help='batch size')
    parser.add_argument("--train_val", type=bool, default=True, help='train with val')
    parser.add_argument("--print_freq", type=int, default=2)
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

    if args.split == 'train':
        image_list = sorted(glob(os.path.join(args.mvtec_ad, args.category, 'train', 'good', '*.png')))
        train_image_list, val_image_list = train_test_split(image_list, test_size=0.2, random_state=0)
        train_dataset = MVTecDataset(train_image_list, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_dataset = MVTecDataset(val_image_list, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        if args.train_val:
            test_neg_image_list = sorted(glob(os.path.join(args.mvtec_ad, args.category, 'test', 'good', '*.png')))
            test_pos_image_list = set(glob(os.path.join(args.mvtec_ad, args.category, 'test', '*', '*.png'))) - set(
                test_neg_image_list)
            test_pos_image_list = sorted(list(test_pos_image_list))
            test_neg_dataset = MVTecDataset(test_neg_image_list, transform=transform)
            test_pos_dataset = MVTecDataset(test_pos_image_list, transform=transform)
            test_neg_loader = DataLoader(test_neg_dataset, batch_size=1, shuffle=False, drop_last=False)
            test_pos_loader = DataLoader(test_pos_dataset, batch_size=1, shuffle=False, drop_last=False)
    elif args.split == 'test':
        test_neg_image_list = sorted(glob(os.path.join(args.mvtec_ad, args.category, 'test', 'good', '*.png')))
        test_pos_image_list = set(glob(os.path.join(args.mvtec_ad, args.category, 'test', '*', '*.png'))) - set(test_neg_image_list)
        test_pos_image_list = sorted(list(test_pos_image_list))
        test_neg_dataset = MVTecDataset(test_neg_image_list, transform=transform)
        test_pos_dataset = MVTecDataset(test_pos_image_list, transform=transform)
        test_neg_loader = DataLoader(test_neg_dataset, batch_size=1, shuffle=False, drop_last=False)
        test_pos_loader = DataLoader(test_pos_dataset, batch_size=1, shuffle=False, drop_last=False)


    if args.split == 'train':
        if args.train_val:
            train_val(teacher, student, train_loader, val_loader, args,  test_pos_loader, test_neg_loader)
        else:
            train_val(teacher, student, train_loader, val_loader, args)
    elif args.split == 'test':
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

        pro = evaluate(gt_pixel, scores, metric='pro')
        auc_pixel = evaluate(gt_pixel.flatten(), scores.flatten(), metric='roc')
        auc_image_max = evaluate(gt_image, scores.max(-1).max(-1), metric='roc')
        print('Catergory: {:s}\tPixel-AUC: {:.6f}\tImage-AUC: {:.6f}\tPRO: {:.6f}'.format(category, auc_pixel, auc_image_max, pro))
     

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
    

def train_val(teacher, student, train_loader, val_loader, args, test_pos_loader=None, test_neg_loader=None):
    min_err = 10000
    teacher.eval()
    student.train()
    optimizer = optim.Momentum(parameters=student.parameters(),
                               learning_rate=args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        student.train()
        epoch_begin = time.time()
        end_time = time.time()
        for index, batch_data in enumerate(train_loader):
            optimizer.clear_grad()
            batch_begin = time.time()
            data_time = batch_begin - end_time
            _, batch_img = batch_data

            with paddle.no_grad():
                t_feat = teacher(batch_img)
            s_feat = student(batch_img)

            loss = paddle.to_tensor(0.0)
            loss.stop_gradient = False
            for i in range(len(t_feat)):
                t_feat[i] = F.normalize(t_feat[i], axis=1)
                s_feat[i] = F.normalize(s_feat[i], axis=1)
                loss += paddle.sum((t_feat[i] - s_feat[i]) ** 2, 1).mean()

            loss.backward()
            optimizer.step()
            lr = optimizer.get_lr()
            end_time = time.time()
            bacth_time = end_time - batch_begin

            if index % args.print_freq == 0:
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t'+
                    "Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, batch time:{:.4f}, data time:{:.4f}".format(
                    epoch,
                    args.batch_size * (index + 1),
                    len(train_loader.dataset),
                    loss.cpu().numpy()[0],
                    float(lr),
                    float(bacth_time),
                    float(data_time)
                ))
        t = time.time() - epoch_begin
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t'+
              "Epoch {} training ends, total {:.2f}s".format(epoch, t))
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t'+
              "Epoch {} testing start".format(epoch))
        err = test(teacher, student, val_loader).mean()
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t'+
              'Valid Loss: {:.7f}'.format(err.item()))
        if err < min_err:
            min_err = err
            save_name = os.path.join(args.model_save_path, args.category, 'best.pdparams')
            dir_name = os.path.dirname(save_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            paddle.save(student.state_dict(), save_name)
        if args.train_val:
            val(args, student, teacher, test_pos_loader, test_neg_loader, epoch)
        t = time.time() - epoch_begin - t
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t'+
              "Epoch {} testing end, total {:.2f}s".format(epoch, t))


def val(args, student, teacher, test_pos_loader, test_neg_loader, epoch, eval_pro=False):
    category = args.category
    gt = load_gt(args.mvtec_ad, category)
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

    if eval_pro: ## very slow
        pro = evaluate(gt_pixel, scores, metric='pro')
        auc_pixel = evaluate(gt_pixel.flatten(), scores.flatten(), metric='roc')
        auc_image_max = evaluate(gt_image, scores.max(-1).max(-1), metric='roc')
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t'+
          'Epoch: {}\tCatergory: {:s}\tPixel-AUC: {:.6f}'
          '\tImage-AUC: {:.6f}\tPRO: {:.6f}'.format(epoch, category, auc_pixel,auc_image_max, pro))
    else:
        auc_pixel = evaluate(gt_pixel.flatten(), scores.flatten(), metric='roc')
        auc_image_max = evaluate(gt_image, scores.max(-1).max(-1), metric='roc')
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t'+
          'Epoch: {}\tCatergory: {:s}\tPixel-AUC: {:.6f}'
          '\tImage-AUC: {:.6f}'.format(epoch, category, auc_pixel,auc_image_max))



if __name__ == "__main__":
    main()
