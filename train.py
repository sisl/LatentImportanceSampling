# packages
import argparse
import os
import pandas as pd
import pickle
import shutil
import tqdm
import yaml

# torch imports
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# nflows imports
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal

# file imports
from construct_transform import create_transform

# set seed for reproducibility
torch.manual_seed(0)

# setup
parser = argparse.ArgumentParser()
parser.add_argument('--simulator',
                    choices=['robot', 'racecar', 'f16'],
                    default='robot',
                    help='Choose an autonomous systems simulator.')
simulator = parser.parse_args()

with open('configs/{}.yaml'.format(simulator.simulator), 'r') as file:
    args = yaml.safe_load(file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#*******************************************************************************
# file IO
#*******************************************************************************
train_df = pd.read_csv("data/{}-flow.csv".format(args['key']), header=None)
train_data = torch.tensor(train_df.values, dtype=torch.float32)

train_dataloader = DataLoader(TensorDataset(train_data), 
    batch_size=args['batch_size'], shuffle=True, pin_memory=True, 
    drop_last=True)


#*******************************************************************************
# flow construction
#*******************************************************************************
base_dist = StandardNormal(shape=[args['features']])
transform = create_transform(args)
flow = Flow(transform, base_dist)
flow = flow.to(device)

optimizer = torch.optim.Adam(flow.parameters(), lr=args['learning_rate'])

total_steps = len(train_dataloader) * args['epochs']


#*******************************************************************************
# flow training
#*******************************************************************************
tb_key = args['base'] + '-' + args['linear'] + '-' + args['key'] + "test"
if os.path.isdir("runs/" + tb_key):
    shutil.rmtree("runs/" + tb_key)
writer = SummaryWriter("runs/" + tb_key)

i = 0
print("training flow...")
pbar = tqdm.tqdm(total=total_steps)
for epoch in range(args['epochs']):
    for x in train_dataloader:
        x = x[0].to(device)

        flow.train()
        optimizer.zero_grad()
        loss = -flow.log_prob(inputs = x).mean()
        i += 1
        writer.add_scalar("Loss/train", loss, i)
        loss.backward()
        if args['grad_norm_clip_value'] is not None:
            clip_grad_norm_(flow.parameters(), args['grad_norm_clip_value'])
        optimizer.step()

        if (i+1) % 50 == 0:
            writer.flush()
        pbar.update(1)

# save flow model
flow.to('cpu')
name = str(tb_key)
f = open("flows/" + name, "wb")
pickle.dump(flow, f)
f.close()
