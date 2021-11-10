# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import skimage.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import vision_transformer as vits

def dino_energy_map(img):
    r, c, _ = img.shape
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = vits.__dict__['vit_small'](patch_size=16, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    state_dict = torch.load('dino_deitsmall16_pretrain.pth', map_location="cpu")
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % 16, img.shape[2] - img.shape[2] % 16
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // 16
    h_featmap = img.shape[-1] // 16

    attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), size=(r, c), mode="nearest")[0].cpu().numpy()

    return attentions[2]