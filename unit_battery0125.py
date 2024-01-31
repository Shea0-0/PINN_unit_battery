import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def min_max(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    return min_val, max_val, max_val-min_val

cp_data = pd.read_parquet('./cp2.parquet')
ic_bc_data = pd.read_parquet('./ic_bc2.parquet')
all_data = pd.concat([cp_data, ic_bc_data])

all_position = all_data[['X', 'Y', 'Z']].drop_duplicates()
cp_position = cp_data[['X', 'Y', 'Z']].drop_duplicates()

# Set default dtype to float32
torch.set_default_dtype(torch.float)
# PyTorch random number generator
torch.manual_seed(1234)
# Random number generators in other libraries
np.random.seed(1234)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_test = torch.tensor(all_data[['X']].values, dtype=torch.float32)
Y_test = torch.tensor(all_data[['Y']].values, dtype=torch.float32)
Z_test = torch.tensor(all_data[['Z']].values, dtype=torch.float32)
time_test = torch.tensor(all_data[['time']].values, dtype=torch.float32)
Temp_test = torch.tensor(all_data[['Temp']].values, dtype=torch.float32)
total_heat_test = torch.tensor(all_data[['total_heat']].values, dtype=torch.float32)

X_CP = torch.tensor(cp_data[['X']].values, dtype=torch.float32)
Y_CP = torch.tensor(cp_data[['Y']].values, dtype=torch.float32)
Z_CP = torch.tensor(cp_data[['Z']].values, dtype=torch.float32)
time_CP = torch.tensor(cp_data[['time']].values, dtype=torch.float32)
Temp_CP = torch.tensor(cp_data[['Temp']].values, dtype=torch.float32)
total_heat_CP = torch.tensor(cp_data[['total_heat']].values, dtype=torch.float32)

# 此处X_IB指cell的IC BC,ic_bc_data包含了所有的tab和cell的ICBC，而CP为cell非ICBC的部分
X_IB = torch.tensor(ic_bc_data[ic_bc_data['is_tab'] == 0][['X']].values, dtype=torch.float32)
Y_IB = torch.tensor(ic_bc_data[ic_bc_data['is_tab'] == 0][['Y']].values, dtype=torch.float32)
Z_IB = torch.tensor(ic_bc_data[ic_bc_data['is_tab'] == 0][['Z']].values, dtype=torch.float32)
time_IB = torch.tensor(ic_bc_data[ic_bc_data['is_tab'] == 0][['time']].values, dtype=torch.float32)
Temp_IB = torch.tensor(ic_bc_data[ic_bc_data['is_tab'] == 0][['Temp']].values, dtype=torch.float32)
total_heat_IB = torch.tensor(ic_bc_data[ic_bc_data['is_tab'] == 0][['total_heat']].values, dtype=torch.float32)

X_tab = torch.tensor(ic_bc_data[ic_bc_data['is_tab'] == 1][['X']].values, dtype=torch.float32)
Y_tab = torch.tensor(ic_bc_data[ic_bc_data['is_tab'] == 1][['Y']].values, dtype=torch.float32)
Z_tab = torch.tensor(ic_bc_data[ic_bc_data['is_tab'] == 1][['Z']].values, dtype=torch.float32)
time_tab = torch.tensor(ic_bc_data[ic_bc_data['is_tab'] == 1][['time']].values, dtype=torch.float32)
Temp_tab = torch.tensor(ic_bc_data[ic_bc_data['is_tab'] == 1][['Temp']].values, dtype=torch.float32)
total_heat_tab = torch.tensor(ic_bc_data[ic_bc_data['is_tab'] == 1][['total_heat']].values, dtype=torch.float32)

# Nu对应ICBC，Nf对应CP
Nu_cell = int(X_IB.shape[0] / 50)
Nu_tab = int(X_tab.shape[0] / 50)
idx_Nu_cell = np.sort(np.random.choice(X_IB.shape[0], Nu_cell, replace=False))
idx_Nu_tab = np.sort(np.random.choice(X_tab.shape[0], Nu_tab, replace=False))
X_Nu_cell = X_IB[idx_Nu_cell, :].float()  # Training Points  of x at (IC+BC)
X_Nu_tab = X_tab[idx_Nu_tab, :].float()  # Training Points  of x at (IC+BC)
time_Nu_cell = time_IB[idx_Nu_cell, :].float()  # Training Points  of t at (IC+BC)
time_Nu_tab = time_tab[idx_Nu_tab, :].float()  # Training Points  of t at (IC+BC)
Y_Nu_cell = Y_IB[idx_Nu_cell, :].float()  # Training Points  of y at (IC+BC)
Y_Nu_tab = Y_tab[idx_Nu_tab, :].float()  # Training Points  of y at (IC+BC)
Z_Nu_cell = Z_IB[idx_Nu_cell, :].float()  # Training Points  of y at (IC+BC)
Z_Nu_tab = Z_tab[idx_Nu_tab, :].float()  # Training Points  of y at (IC+BC)
total_heat_Nu_cell = total_heat_IB[idx_Nu_cell, :].float()  # Training Points  of y at (IC+BC)
total_heat_Nu_tab = total_heat_tab[idx_Nu_tab, :].float()  # Training Points  of y at (IC+BC)
Temp_Nu_cell = Temp_IB[idx_Nu_cell, :].float()
Temp_Nu_tab = Temp_tab[idx_Nu_tab, :].float()

time_steps = cp_data['time'].nunique()
nodes_cell_num = cp_data[['X', 'Y', 'Z']].drop_duplicates().shape[0]

Nf_cell = 3400  # Nf: Number of collocation points
idx_Nf_cell = np.sort(np.random.choice(nodes_cell_num*time_steps, Nf_cell, replace=False))
X_Nf = X_CP[idx_Nf_cell, :]
Y_Nf = Y_CP[idx_Nf_cell, :]
Z_Nf = Z_CP[idx_Nf_cell, :]
time_Nf = time_CP[idx_Nf_cell, :]
Temp_Nf = Temp_CP[idx_Nf_cell, :]
total_heat_Nf = total_heat_CP[idx_Nf_cell, :]

# train包含三部分：cell(IC/BC)+cell(CP)+tab,用以在构建loss function时使用不同的cp和rho
X_train = torch.vstack((X_Nu_cell, X_Nf, X_Nu_tab)).float()  # Collocation Points of x (CP)
Y_train = torch.vstack((Y_Nu_cell, Y_Nf, Y_Nu_tab)).float()
Z_train = torch.vstack((Z_Nu_cell, Z_Nf, Z_Nu_tab)).float()
time_train = torch.vstack((time_Nu_cell, time_Nf, time_Nu_tab)).float()
total_heat_train = torch.vstack((total_heat_Nu_cell, total_heat_Nf, total_heat_Nu_tab)).float()
Temp_train = torch.vstack((Temp_Nu_cell, Temp_Nf, Temp_Nu_tab)).float()

from neuromancer.dataset import DictDataset

# turn on gradients for PINN
X_train.requires_grad = True
Y_train.requires_grad = True
Z_train.requires_grad = True
time_train.requires_grad = True
Temp_train.requires_grad = True

# Training dataset
train_data = DictDataset({'t': time_train, 'y': Y_train, 'x': X_train, 'z': Z_train}, name='train')
# test dataset
test_data = DictDataset({'t': time_test, 'y': Y_test, 'x': X_test, 'z': Z_test,
                         'temperature': Temp_test}, name='test')

# torch dataloaders
batch_size = X_train.shape[0]  # full batch training
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           collate_fn=train_data.collate_fn,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          collate_fn=test_data.collate_fn,
                                          shuffle=False)

from neuromancer.modules import blocks
from neuromancer.system import Node

# neural net to solve the PDE problem bounded in the PDE domain
net = blocks.MLP(insize=4, outsize=1, hsizes=[64, 64], nonlin=nn.Tanh)

# symbolic wrapper of the neural net
pde_net = Node(net, ['t', 'y', 'x', 'z'], ['temperature_hat'], name='net')

print("symbolic inputs  of the pde_net:", pde_net.input_keys)
print("symbolic outputs of the pde_net:", pde_net.output_keys)

# evaluate forward pass on the train data
net_out = pde_net(train_data.datadict)
net_out['temperature_hat'].shape

from neuromancer.constraint import variable

# symbolic Neuromancer variables
temperature_hat = variable('temperature_hat')  # PDE solution generated as the output of a neural net (pde_net)
var_t = variable('t')  # temporal domain
var_y = variable('y')
var_x = variable('x')  # spatial domain
var_z = variable('z')  # spatial domain

# get the symbolic derivatives
dtemperature_dt = (temperature_hat).grad(var_t)
dtemperature_dx = (temperature_hat).grad(var_x)
dtemperature_dy = (temperature_hat).grad(var_y)
dtemperature_dz = (temperature_hat).grad(var_z)
d2temperatur_d2x = dtemperature_dx.grad(var_x)
d2temperatur_d2y = dtemperature_dy.grad(var_y)
d2temperatur_d2z = dtemperature_dz.grad(var_z)
# parameters
rho_cell = 2092
C_p_cell = 678
k_cell = 18.2
rho_tab = 8978
C_p_tab = 381
k_tab = 387.6
# 创建相应的 rho、C_p、k 张量
rho = torch.ones((X_train.shape[0], 1)) * rho_cell
C_p = torch.ones((X_train.shape[0], 1)) * C_p_cell
k = torch.ones((X_train.shape[0], 1)) * k_cell

rho[(Nu_cell + Nf_cell):] = rho_tab
C_p[(Nu_cell + Nf_cell):] = C_p_tab
k[(Nu_cell + Nf_cell):] = k_tab

f_pinn = (rho * C_p * dtemperature_dt - k * d2temperatur_d2x - k * d2temperatur_d2y - k * d2temperatur_d2z
          - total_heat_train)

# check the shapes of the forward pass of the symbolic PINN terms
print(dtemperature_dt({**net_out, **train_data.datadict}).shape)
print(dtemperature_dx({**net_out, **train_data.datadict}).shape)
print(dtemperature_dy({**net_out, **train_data.datadict}).shape)
print(dtemperature_dz({**net_out, **train_data.datadict}).shape)
print(d2temperatur_d2x({**net_out, **train_data.datadict}).shape)
print(d2temperatur_d2y({**net_out, **train_data.datadict}).shape)
print(d2temperatur_d2z({**net_out, **train_data.datadict}).shape)
print(f_pinn({**net_out, **train_data.datadict}).shape)

# computational graph of the PINN neural network
f_pinn.show()
# scaling factor for better convergence
scaling = 100.

# PDE CP loss
ell_f = scaling * (f_pinn == 0.) ^ 2

# PDE IC and BC loss
ell_u1 = scaling * ((temperature_hat[:Nu_cell] == Temp_train[:Nu_cell]) ^ 2)
ell_u2 = scaling * ((temperature_hat[-Nu_tab:] == Temp_train[-Nu_tab:]) ^ 2)

# # output constraints to bound the PINN solution in the PDE output domain [-1.0, 1.0]

con_1 = scaling * (temperature_hat <= min_max(Temp_test)[1]) ^ 2
con_2 = scaling * (temperature_hat >= min_max(Temp_test)[0]) ^ 2

from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem

# create Neuromancer optimization loss
pinn_loss = PenaltyLoss(objectives=[ell_f, ell_u1, ell_u2], constraints=[con_1, con_2])

# construct the PINN optimization problem
problem = Problem(nodes=[pde_net],  # list of nodes (neural nets) to be optimized
                  loss=pinn_loss,  # physics-informed loss function
                  grad_inference=True  # argument for allowing computation of gradients at the inference time)
                  )

from neuromancer.trainer import Trainer

optimizer = torch.optim.Adam(problem.parameters(), lr=0.01)
epochs = 10000

#  Neuromancer trainer
trainer = Trainer(
    problem.to(device),
    train_loader,
    optimizer=optimizer,
    epochs=epochs,
    epoch_verbose=200,
    train_metric='train_loss',
    dev_metric='train_loss',
    eval_metric="train_loss",
    warmup=epochs,
)

# Train PINN
best_model = trainer.train()

# load best trained model
problem.load_state_dict(best_model)

# evaluate trained PINN on test data
PINN = problem.nodes[0]
temperature1 = PINN(test_data.datadict)['temperature_hat']


# arrange data for plotting
temperature_pinn = temperature1.reshape(shape=[time_steps, 388*4]).detach().cpu()

# contour(temperature_pinn[89, :], 1)
#
# contour(temperature_pinn[89, :]-temp[89, :], 1)

print("0")
