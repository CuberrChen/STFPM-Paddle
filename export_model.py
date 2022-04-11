import os
import argparse

import paddle

from models.resnet import ResNet_MS3, ResNet_MS3_EXPORT

def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument("--depth", type=int, default=18, help="resnet depth")
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the exported model',
        type=str,
        default='./output')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for export',
        type=str,
        default=None)
    parser.add_argument(
        '--img_size',
        help='The img size for export',
        type=int,
        default=256)

    return parser.parse_args()


def main(args):

    # build model
    teacher = ResNet_MS3(depth=args.depth, pretrained=True)
    student = ResNet_MS3(depth=args.depth, pretrained=False)

    if args.model_path is not None:
        student.load_dict(paddle.load(args.model_path))
        print('Loaded trained params of model successfully.')
    STFPM = ResNet_MS3_EXPORT(student, teacher)
    shape = [-1, 3, args.img_size, args.img_size]
    new_net = STFPM
    new_net.eval()
    new_net = paddle.jit.to_static(
        new_net,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(new_net, save_path)
    print(f'Model is saved in {args.save_dir}.')


if __name__ == '__main__':
    args = parse_args()
    main(args)