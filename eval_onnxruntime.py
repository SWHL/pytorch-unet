# !/usr/bin/env python
# -*- encoding: utf-8 -*-
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def reverse_transform(inp):
    inp = inp.transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp


def scale_resize(img, resize_value=(480, 480)):
        '''
        @params:
        img: ndarray
        resize_value: (width, height)
        '''
        # padding
        ratio = resize_value[0] / resize_value[1]  # w / h
        h, w = img.shape[:2]
        if w / h < ratio:
            # 补宽
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = cv2.copyMakeBorder(img, 0, 0, w_padding, w_padding,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            # 补高  (left, upper, right, lower)
            t = int(w / ratio)
            h_padding = (t - h) // 2
            color = tuple([int(i) for i in img[0, 0, :]])
            img = cv2.copyMakeBorder(img, h_padding, h_padding, 0, 0,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img = cv2.resize(img, resize_value,
                            interpolation=cv2.INTER_LANCZOS4)
        return img


if __name__ == '__main__':
    onnx_path = 'convert2onnx/2022-05-04-09-36-54-best.onnx'
    ort_session = ort.InferenceSession(onnx_path)

    img_path = 'datasets/gen_datasets/train/images/114.jpg'
    img = cv2.imread(img_path)
    image = scale_resize(img).astype(np.float32)
    image /= 255.0
    image = image.transpose([2, 0, 1])
    image = image[np.newaxis, ...]

    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: image}

    import time
    s = time.time()
    ort_outs = ort_session.run(None, ort_inputs)[0]
    print(f'elapse: {time.time() - s}')

    ort_outs = reverse_transform(ort_outs[0])
    cv2.imwrite(f'tmp/pred_{Path(img_path).name}', ort_outs)


