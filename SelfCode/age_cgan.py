import numpy as np
import torch
from torch import nn

num_epochs = 100
d_steps = 10
g_steps = 10


def get_distribution_sampler(mu, sigma):
	return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))


def get_generator_input_sampler():
	return lambda m, n: torch.rand(m, n)


class Generator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, f):
		super(Generator, self).__init__()
		self.map1 = nn.Linear(input_size, hidden_size)
		self.map2 = nn.Linear(hidden_size, hidden_size)
		self.map3 = nn.Linear(hidden_size, output_size)
		self.f = f

	def forward(self, x):
		x = self.map1(x)
		x = self.f(x)
		x = self.map2(x)
		x = self.f(x)
		x = self.map3(x)
		return x


class Discriminator(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, f):
		super(Discriminator, self).__init__()
		self.map1 = nn.Linear(input_size, hidden_size)
		self.map2 = nn.Linear(hidden_size, hidden_size)
		self.map3 = nn.Linear(hidden_size, output_size)
		self.f = f

	def forward(self, x):
		x = self.map1(x)
		x = self.f(x)
		x = self.map2(x)
		x = self.f(x)
		x = self.map3(x)
		x = self.f(x)
		return x


if __name__ == '__main__':
	D = Discriminator(1, 6, 3, torch.sigmoid)
	D.zero_grad()