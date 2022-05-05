# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: main.py
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from tqdm import tqdm
from decord import VideoReader
from decord import cpu

from PIL import Image, ImageFont, ImageDraw


def read_txt(txt_path):
    with open(txt_path, 'r', encoding='gb2312') as f:
        data = list(map(lambda x: x.strip('\n'), f))
    return data


def mkdir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


class GenSubtitle(object):
    def __init__(self, font_path, font_size=36,
                 text_color=(0, 0, 0), bg_color=(255, 255, 255)):
        self.font = ImageFont.truetype(font_path, font_size)
        self.text_color = text_color
        self.bg_color = bg_color

    def __call__(self, img, point, text):
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        mask = Image.new('RGB', img.size, color=self.text_color)
        draw_mask = ImageDraw.Draw(mask)

        draw.text(point, text, fillcolor=self.bg_color, font=self.font)
        draw_mask.text(point, text, fillcolor=(255, 255, 255), font=self.font)
        return img, mask


if __name__ == '__main__':
    save_img_dir = Path('gen_datasets/train/images')
    mkdir(save_img_dir)

    save_mask_dir = Path('gen_datasets/train/masks')
    mkdir(save_mask_dir)

    font_path = 'assets/fonts/simhei.ttf'
    gen = GenSubtitle(font_path)

    # 插入字幕位置，文本左上角位置坐标
    point = [438, 58]

    # 生成图像高度
    crop_h = 152

    font_size = 36

    video_path = 'assets/mp4/The.Matrix.Reloaded/The.Matrix.Reloaded.mkv'
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = int(vr.get_avg_fps())

    corpus = read_txt('assets/mp4/The.Matrix.Reloaded/The.Matrix.Reloaded.srt')

    for index in tqdm(range(len(corpus))):
        img = vr[index * fps].asnumpy()

        h, w = img.shape[:2]
        img = img[h-crop_h:, :, :]

        img, mask = gen(img, point, corpus[index], font_size)

        img.save(f'{save_img_dir}/{index}.jpg')
        mask.save(f'{save_mask_dir}/{index}.jpg')
