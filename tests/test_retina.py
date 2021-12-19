import sys
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image

sys.path.append('.')
print(sys.path)

from agent.stubs.retina import Retina

import cv2

config = {
  'f_size': 7,
  'f_sigma': 2.0,
  'f_k': 1.6,  # approximates Laplacian of Gaussian
  'summaries': True
}

channels = 3
model = Retina("retina-test", channels=channels, config=config)

file_path = sys.argv[1]
print('Loading image: ', file_path)

# OpenCV
# array([[[255,  23,  23],
# (1598, 1598, 3)
img_cv = cv2.imread(file_path)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img_cv)
cv2.waitKey(0)  # 何かキーを押すと画面を閉じる
cv2.destroyAllWindows()

# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1598x1598>
img = Image.open(file_path)
if channels == 1:
  img = img.convert("L")
print(img.size)

# tensor([[[0.0902, 0.0000, 0.0039,  ..., 0.0000, 0.0000, 0.0000],
img_tensor = torchvision.transforms.ToTensor()(img)
# torch.Size([3, 1598, 1598])
img_tensor_shape = img_tensor.shape
# tensor([[[[0.0902, 0.0000, 0.0039,  ..., 0.0000, 0.0000, 0.0000],
# torch.Size([1, 3, 1598, 1598])
img_tensor = torch.unsqueeze(img_tensor, 0)  # insert batch dimensions

"""
nn.Module.__call__(x) ===> forward(x)   順伝播
"""
# tensor([[[[-3.9786e-03,  1.6002e-04, -5.6324e-05,  ...,  0.0000e+00,
# torch.Size([1, 6, 1592, 1592])
# tensor([[[[-3.9786e-03,  1.6002e-04, -5.6324e-05,  ...,  0.0000e+00,
# torch.Size([1, 3, 1592, 1592])
# tensor([[[[ 3.9786e-03, -1.6002e-04,  5.6324e-05,  ...,  0.0000e+00,
# torch.Size([1, 3, 1592, 1592])
dog, dog_pos_tensor, dog_neg_tensor = model(img_tensor)

print('dog shape', dog_pos_tensor.shape)
# remove batch and channel dimensions
# tensor([[[-3.9786e-03,  1.6002e-04, -5.6324e-05,  ...,  0.0000e+00,
# torch.Size([3, 1592, 1592])
dog_pos_tensor = torch.squeeze(dog_pos_tensor)
# tensor([[[ 3.9786e-03, -1.6002e-04,  5.6324e-05,  ...,  0.0000e+00,
# torch.Size([3, 1592, 1592])
dog_neg_tensor = torch.squeeze(dog_neg_tensor)

# <PIL.Image.Image image mode=RGB size=1592x1592>
dog_pos = torchvision.transforms.ToPILImage()(dog_pos_tensor)
# <PIL.Image.Image image mode=RGB size=1592x1592>
dog_neg = torchvision.transforms.ToPILImage()(dog_neg_tensor)
print('dog shape squeezed', dog_pos_tensor.shape)
print(dog_pos.size)
print(dog_neg.size)

# show with PIL
img.show()
dog_pos.show()
dog_neg.show()

# show in matplotlib figure   --> currently has weird colours, maybe b/c range [0, 256] instead of [-1,1]
fig = plt.figure()

ax = fig.add_subplot(1, 3, 1)
imgplot = plt.imshow(img)
ax.set_title('Original')

ax = fig.add_subplot(1, 3, 2)
imgplot = plt.imshow(dog_pos)
ax.set_title('DoG+')

ax = fig.add_subplot(1, 3, 3)
imgplot = plt.imshow(dog_neg)
ax.set_title('DoG-')

plt.show()
plt.savefig('results.png')
