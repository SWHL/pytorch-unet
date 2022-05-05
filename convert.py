# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: conver2onnx.py
# @Time: 2020/11/27 17:53:05
# @Author: Max
from collections import OrderedDict
import pytorch_unet
import numpy as np
import onnx
import onnxruntime
import torch.onnx


def to_numpy(data):
    if data.requires_grad:
        return data.detach().cpu().numpy()
    else:
        return data.cpu().numpy()


save_onnx_path = 'convert2onnx/2022-05-04-09-36-54-best.onnx'
weight_path = 'exp/2022-05-04-09-36-54/best.pth'

model = pytorch_unet.ResNetUNet(1)
state_dict = torch.load(weight_path, map_location='cpu')
new_state_dict = OrderedDict()
for key, value in state_dict.items():
    key = key.replace('module.', '')
    new_state_dict[key] = value
model.load_state_dict(new_state_dict)
model.eval()

x = torch.randn(1, 3, 480, 480)
with torch.no_grad():
    torch_out = model(x)

# Export the model
torch.onnx.export(model,
                  x,
                  save_onnx_path,
                  export_params=True,
                  opset_version=12,
                  verbose=False,
                  input_names=['input'],
                  output_names=['output'],
                  do_constant_folding=True,
                  dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'},
                                'output': {0: 'batch', 2: 'height', 3: 'width'}
                                }
                  )

# x = np.random.randn(1, 3, 32, 160).astype(np.float32)
ort_session = onnxruntime.InferenceSession(save_onnx_path)

input_name = ort_session.get_inputs()[0].name
ort_inputs = {input_name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-3, atol=1e-5)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")