import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from datasets import load_dataset
from torch import multiprocessing
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification


class IMDB_Dataset(Dataset):
	def __init__(self):
		self.dataset = load_dataset("imdb")['train']

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, item):
		instance = self.dataset[item]
		return instance['text'], instance['label']


def init_seeds(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	cudnn.deterministic = True
	cudnn.benchmark = False


def ddp_group_setup(rank, world_size):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12399'
	dist.init_process_group("nccl", rank=rank, world_size=world_size)


def ddp_group_cleanup():
	dist.destroy_process_group()


def training_basic(rank, world_size):
	init_seeds(42)
	ddp_group_setup(rank, world_size)
	device = torch.device('cuda', rank)

	config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
	model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
	model = model.to(device)
	ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in ddp_model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": 0.01,
		},
		{
			"params": [p for n, p in ddp_model.named_parameters() if any(nd in n for nd in no_decay)],
			"weight_decay": 0.0,
		},
	]

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	train_dataset = IMDB_Dataset()
	train_sampler = DistributedSampler(train_dataset, shuffle=True)
	train_loader = DataLoader(train_dataset, batch_size=32, drop_last=False, pin_memory=True, sampler=train_sampler)

	optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=0.00001)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader), gamma=0.9)

	epoch = 0
	while True:
		ddp_model.train()
		train_loader.sampler.set_epoch(epoch)
		total_loss = 0
		for step, (data, label) in enumerate(train_loader):
			encoded_data = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
			encoded_data, label = encoded_data.to(device), label.to(device)
			outputs = ddp_model(**encoded_data, labels=label)
			loss = outputs.loss
			total_loss += loss.item()

			loss.backward()
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()

		if rank == 0:
			print(f"epoch : {epoch}, total loss : {total_loss / len(train_loader)}")

		if rank == 0 and (epoch + 1) % 1000000 == 0:
			if not os.path.exists(f"./checkpoints/checkpoint-{epoch}"):
				os.makedirs(f"./checkpoints/checkpoint-{epoch}")
			torch.save(ddp_model.module.state_dict(), f"./checkpoints/checkpoint-{epoch}/model.pth")
			torch.save(optimizer.state_dict(), f"./checkpoints/checkpoint-{epoch}/optimizer.pth")
			training_state = {'epoch': epoch}
			torch.save(training_state, f"./checkpoints/checkpoint-{epoch}/training_state.pth")

		epoch += 1

	ddp_group_cleanup()


if __name__ == "__main__":
	n_gpus = torch.cuda.device_count()
	world_size = n_gpus
	multiprocessing.spawn(training_basic, args=(world_size,), nprocs=world_size, join=True)
