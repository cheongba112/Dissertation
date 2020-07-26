import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19 as v19

import numpy as np


class VGG19(nn.Module):
	def __init__(self):
		super(VGG19, self).__init__()
		self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1, stride=1)
		self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
		self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
		self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
		self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
		self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
		self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
		self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
		self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
		self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
		self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
		self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
		self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
		self.fc1 = nn.Linear(7 * 7 * 512, 4096)
		self.fc2 = nn.Linear(4096, 4096)
		self.fc3 = nn.Linear(4096, 1)

	def forward(self, x):
		x = F.relu(self.conv1_1(x))
		x = F.relu(self.conv1_2(x))
		x = F.max_pool2d(x, 2)
		x = F.relu(self.conv2_1(x))
		x = F.relu(self.conv2_2(x))
		x = F.max_pool2d(x, 2)
		x = F.relu(self.conv3_1(x))
		x = F.relu(self.conv3_2(x))
		x = F.relu(self.conv3_3(x))
		x = F.max_pool2d(x, 2)
		x = F.relu(self.conv4_1(x))
		x = F.relu(self.conv4_2(x))
		x = F.relu(self.conv4_3(x))
		x = F.max_pool2d(x, 2)
		x = F.relu(self.conv5_1(x))
		x = F.relu(self.conv5_2(x))
		x = F.relu(self.conv5_3(x))
		x = F.max_pool2d(x, 2)
		x = x.view(x.size(0), -1)  # Flatten
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


# Transfer between images and tensors
loader = transforms.Compose([ transforms.ToTensor() ])
unloader = transforms.ToPILImage()


has_cuda = torch.cuda.is_available()


def image2tensor(image_path):
	image = Image.open(image_path)
	image = image.resize((224, 224), Image.BILINEAR)
	image = image.convert('RGB')
	image = loader(image).unsqueeze(0)
	tensor = image.to(torch.float)
	return tensor


if __name__ == '__main__':
	# vgg19 = v19(num_classes=1)
	vgg19 = VGG19()

	optimizer = torch.optim.SGD(vgg19.parameters(), lr=0.01)
	loss_func = nn.MSELoss()

	x = image2tensor('sample.jpg')
	y = torch.tensor(10.0)

	epoch_num = 10

	for i in range(epoch_num):
		pred = vgg19(x)
		loss = loss_func(pred, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(pred, loss)

	# print(torch.cuda.is_available())

		# if not i % 100:
			# print(loss)

	if has_cuda:
		pass

	# print(image2tensor('sample.jpg').size())
	
	# net = Net()
	# out = net(image2tensor('sample.jpg'))
	# print(out.size())
	
	# print(torch.Tensor([[1], [2]]).size())
	# print(np.ones([3, 3]) * 3)



# how to save and load pre-trained models (比如加一个vgg作为一部分，后面加自己的东西) (.pkl .pt .pth)
# how to use GPU for training
# how to use dataloader to prepare dataset

