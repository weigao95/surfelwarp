import cv2
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('output', help='output file')
    parser.add_argument('frame_num', type=int, help='number of generated frames')
    args = parser.parse_args()
    return args

def main(args):
    for i in range(args.frame_num):
        # 创建一个 640x480 的纯白图像
        height, width = 440, 600
        white_mask = np.ones((height, width), dtype=np.uint8) * 255

        # 保存图像
        cv2.imwrite(args.output + f"/frame-{i:06d}.mask.png", white_mask)

if __name__ == '__main__':
    args = parse_args()
    main(args)