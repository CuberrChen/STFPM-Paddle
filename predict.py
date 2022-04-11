import argparse
import cv2
import numpy as np
from PIL import Image
import paddle
import paddle.nn.functional as F
from paddle.vision import transforms

from models.resnet import ResNet_MS3


def main():
    parser = argparse.ArgumentParser(description="Anomaly Detection")
    # required training super-parameters
    parser.add_argument("--depth", type=int, default=18, help="student resnet depth")
    parser.add_argument("--checkpoint", type=str, default=None, help="student checkpoint")
    # trivial parameters
    parser.add_argument("--image_path", type=str, default=None, help="picture path")
    parser.add_argument("--save_path", type=str, default='results', help="save results")

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

    saved_dict = paddle.load(args.checkpoint)
    print('load ' + args.checkpoint)
    student.load_dict(saved_dict)
    output = predict(teacher, student, args.image_path, transform)

    # Show or save the output image, depending on what's provided as
    # the command line argument.
    output = anomaly_map_to_color_map(output, True)
    if args.save_path is None:
        cv2.imshow("Anomaly Map", output)
    else:
        cv2.imwrite(filename=str(args.save_path), img=output)

def predict(teacher, student, img_path, transform):
    img = Image.open(img_path).convert('RGB')
    orishape = img.size
    img = transform(img)
    img = img.unsqueeze(0)
    teacher.eval()
    student.eval()
    with paddle.no_grad():
        t_feat = teacher(img)
        s_feat = student(img)
    score_map = 1.
    for j in range(len(t_feat)):
        t_feat[j] = F.normalize(t_feat[j], axis=1)
        s_feat[j] = F.normalize(s_feat[j], axis=1)
        sm = paddle.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
        sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)
        # aggregate score map by element-wise product
        score_map = score_map * sm # layer map
    score_map = score_map.squeeze().cpu().numpy()
    anomaly_map = cv2.resize(score_map, (orishape[0],orishape[1]))
    return anomaly_map

def anomaly_map_to_color_map(anomaly_map: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute anomaly color heatmap.
    Args:
        anomaly_map (np.ndarray): Final anomaly map computed by the distance metric.
        normalize (bool, optional): Bool to normalize the anomaly map prior to applying
            the color map. Defaults to True.
    Returns:
        np.ndarray: [description]
    """
    if normalize:
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
    anomaly_map = anomaly_map * 255
    anomaly_map = anomaly_map.astype(np.uint8)

    anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    anomaly_map = cv2.cvtColor(anomaly_map, cv2.COLOR_BGR2RGB)
    return anomaly_map

if __name__ == "__main__":
    main()

paddle.nn.functional.normalize()