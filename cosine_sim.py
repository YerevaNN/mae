import os
import torch 
from pl_train import LightningMAE

from dataset import MAEDataset
from torchmetrics.functional import pairwise_cosine_similarity as cos_dist

import models_mae


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()) #.clamp(min=eps)


BATCH_SIZE = 1
arch='mae_vit_large_patch16'
model_mae = getattr(models_mae, arch)()

chkpt_dir = '/mnt/2tb/alla/mae/mae_contastive/lightning_logs/version_12/checkpoints/epoch=30-step=31.ckpt'
chkpt_dir_old = '/mnt/2tb/hrant/checkpoints/mae_models/mae_visualize_vit_large.pth'
checkpoint = torch.load(chkpt_dir_old, map_location='cpu')
msg = model_mae.load_state_dict(checkpoint['model'], strict=False)
model_mae = LightningMAE.load_from_checkpoint(chkpt_dir, model=model_mae)

model_mae.eval()

root = '/mnt/2tb/hrant/FAIR1M/fair1m_1000/train1000/'
path_ann = os.path.join(root, 'few_shot_8.json')
path_imgs = os.path.join(root, 'images')
dataset = MAEDataset(path_ann, path_imgs, resize_image=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

dl = next(iter(dataloader))
img = torch.einsum('nhwc->nchw', dl['image'])
img_enc = model_mae(img.float())
img_enc = img_enc.reshape(-1, img_enc.shape[-1])

cos_torchmetrics = cos_dist(img_enc, img_enc)
cos_custom = cosine_distance_torch(img_enc)

print((cos_torchmetrics.reshape(-1) != cos_custom.reshape(-1)).sum())
ind = cos_torchmetrics != cos_custom
print(cos_torchmetrics[ind] , cos_custom[ind])
print((cos_torchmetrics.reshape(-1).abs() - cos_custom.reshape(-1).abs()).sum())