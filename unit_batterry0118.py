# torch and numpy imports
import numpy
import torch
import torch.nn as nn
import numpy as np
# plotting imports
import os
import matplotlib.pyplot as plt

# z=1,2,3,4，表示z方向的4层节点
def contour(temp_c, z):
    x = np.linspace(-0.0725, 0.0725, 17)
    y = np.linspace(-0.096, 0.141, 24)
    x_mesh, y_mesh = np.meshgrid(x, y)
    temperature = np.zeros_like(x_mesh, dtype=float)
    temp_c = temp_c[(z-1)*388:z*388]
    index_c = 0
    for i_c in range(24):
        if i_c < 20:
            for j_c in range(17):
                temperature[i_c, j_c] = temp_c[index_c]
                index_c += 1
        if i_c > 19:
            for j_c in range(17):
                if j_c in {0, 7, 8, 9, 16}:
                    temperature[i_c, j_c] = np.nan
                else:
                    temperature[i_c, j_c] = temp_c[index_c]
                    index_c += 1
    plt.figure(figsize=(14.5, 23.7))
    plt.contourf(x, y, temperature, levels=1000, cmap='jet')
    plt.colorbar(label='T')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('T')
    plt.show()


def file_path(path):
    file_paths = []
    file_num = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if not file_extension:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                file_num += 1
    return file_paths, file_num

def min_max(data):
    min_val = torch.min(data)
    max_val = torch.max(data)
    return min_val, max_val, max_val-min_val

# Set default dtype to float32
torch.set_default_dtype(torch.float)
# PyTorch random number generator
torch.manual_seed(1234)
# Random number generators in other libraries
np.random.seed(1234)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_paths, file_num = file_path("E:/desktop/ansys/data20231208noflux/data20231208")
# 初始化从文件中提取的数据
file_x = torch.empty(0)
file_y = torch.empty(0)
file_z = torch.empty(0)
file_temp = torch.empty(0)
file_total_heat = torch.empty(0)


# 遍历列表，逐个处理文件
for file_path in file_paths:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = lines[1:]
        # 初始化空列表来存储处理后的数据
        processed_data = []

        for line in data:
            parts = line.strip().split()
            processed_line = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
            processed_data.append(processed_line)
        tensor_data = torch.tensor(processed_data)
        file_x = torch.cat((file_x, tensor_data[:, 0]), dim=0)
        file_y = torch.cat((file_y, tensor_data[:, 1]), dim=0)
        file_z = torch.cat((file_z, tensor_data[:, 2]), dim=0)
        file_temp = torch.cat((file_temp, tensor_data[:, 3]), dim=0)
        file_total_heat = torch.cat((file_total_heat, tensor_data[:, 4]), dim=0)

# 构建张量
nodes_num = len(data)
nodes = torch.linspace(1, nodes_num, nodes_num)
t = torch.linspace(0, 30 * (file_num-1), file_num)
T, N = torch.meshgrid(t, nodes, indexing='ij')
X = torch.empty_like(N)
Y = torch.empty_like(N)
Z = torch.empty_like(N)
temp = torch.empty_like(N)
total_heat = torch.empty_like(N)

# 使用循环填入张量
index = 0  # 初始化索引
for i in range(file_num):
    for j in range(nodes_num):
        X[i, j] = file_x[index]
        Y[i, j] = file_y[index]
        Z[i, j] = file_z[index]
        temp[i, j] = file_temp[index]
        total_heat[i, j] = file_total_heat[index]
        index += 1


contour(temp[89, :], 1)

# testdata
X_test = X.flatten()[:, None].float()  # the input dataset of x
T_test = T.flatten()[:, None].float()  # the input dataset of t
Y_test = Y.flatten()[:, None].float()  # the input dataset of y
Z_test = Z.flatten()[:, None].float()  # the input dataset of y
temp_test = temp.flatten()[:, None].float()
total_heat_test = total_heat.flatten()[:, None].float()


# boundary condition
bottom_x_index = np.array([], dtype=int)
bottom_y_index = np.array([], dtype=int)
bottom_z_index = np.array([], dtype=int)
top_x_index = np.array([], dtype=int)
top_y_index = np.array([], dtype=int)
top_z_index = np.array([], dtype=int)
tab_index = np.array([], dtype=int)
for i in range(4):
    bottom_x_index = np.append(bottom_x_index, np.arange(i * 388, i * 388 + 20 * 17, 17))
    bottom_y_index = np.append(bottom_y_index, np.arange(i * 388, i * 388 + 17, 1))
    top_x_index = np.append(top_x_index, np.arange(i * 388 + 16, i * 388 + 20 * 17, 17))
    top_y_index = np.append(top_y_index, np.arange(i * 388 + 323, i * 388 + 340, 1))
    tab_index = np.append(tab_index, np.arange(i * 388 + 340, i * 388 + 388, 1))
bottom_z_index = np.arange(0, 388, 1)
top_z_index = np.arange(3*388, 4*388, 1)

BC_index = np.concatenate((bottom_x_index, bottom_y_index,bottom_z_index,top_x_index,top_y_index,top_z_index,tab_index))


# initial condition
initial_X = X[0, :].reshape(-1, 1)
initial_Y = Y[0, :].reshape(-1, 1)
initial_T = T[0, :].reshape(-1, 1)
initial_Z = Z[0, :].reshape(-1, 1)
initial_temp = temp[0, :].reshape(-1, 1)
initial_total_heat = total_heat[0, :].reshape(-1, 1)

BC_X = X[:, BC_index].reshape(-1, 1)
BC_Y = Y[:, BC_index].reshape(-1, 1)
BC_T = T[:, BC_index].reshape(-1, 1)
BC_Z = Z[:, BC_index].reshape(-1, 1)
BC_temp = temp[:, BC_index].reshape(-1, 1)
BC_total_heat = total_heat[:, BC_index].reshape(-1, 1)

# train data
X_train = torch.vstack([initial_X, BC_X])
T_train = torch.vstack([initial_T, BC_T])
Y_train = torch.vstack([initial_Y, BC_Y])
Z_train = torch.vstack([initial_Z, BC_Z])
total_heat_train = torch.vstack([initial_total_heat, BC_total_heat])
temp_train = torch.vstack([initial_temp, BC_temp])

Nu = int(X_train.shape[0] / 50)
idx_Nu = np.sort(np.random.choice(X_train.shape[0], Nu, replace=False))
X_train_Nu = X_train[idx_Nu, :].float()  # Training Points  of x at (IC+BC)
T_train_Nu = T_train[idx_Nu, :].float()  # Training Points  of t at (IC+BC)
Y_train_Nu = Y_train[idx_Nu, :].float()  # Training Points  of y at (IC+BC)
Z_train_Nu = Z_train[idx_Nu, :].float()  # Training Points  of y at (IC+BC)
total_heat_train_Nu = total_heat_train[idx_Nu, :].float()  # Training Points  of y at (IC+BC)
temp_train_Nu = temp_train[idx_Nu, :].float()

#  Choose (Nf) Collocation Points
Nf = 5000  # Nf: Number of collocation points
idx_Nf = np.sort(np.random.choice(nodes_num*file_num, Nf, replace=False))
X_train_CP = X_test[idx_Nf, :].view(-1, 1)
Y_train_CP = Y_test[idx_Nf, :].view(-1, 1)
T_train_CP = T_test[idx_Nf, :].view(-1, 1)
Z_train_CP = Z_test[idx_Nf, :].view(-1, 1)
temp_train_CP = temp_test[idx_Nf, :].view(-1, 1)
total_heat_train_CP = total_heat_test[idx_Nf, :].view(-1, 1)

# add IC+BC to the collocation points
X_train_Nf = torch.vstack((X_train_CP, X_train_Nu)).float()  # Collocation Points of x (CP)
T_train_Nf = torch.vstack((T_train_CP, T_train_Nu)).float()  # Collocation Points of t (CP)
Y_train_Nf = torch.vstack((Y_train_CP, Y_train_Nu)).float()
Z_train_Nf = torch.vstack((Z_train_CP, Z_train_Nu)).float()
total_heat_train_Nf = torch.vstack((total_heat_train_CP, total_heat_train_Nu)).float()
temp_train_Nf = torch.vstack((temp_train_CP, temp_train_Nu)).float()


from neuromancer.dataset import DictDataset

# turn on gradients for PINN
X_train_Nf.requires_grad = True
Y_train_Nf.requires_grad = True
Z_train_Nf.requires_grad = True
T_train_Nf.requires_grad = True
temp_train_Nf.requires_grad = True

# Training dataset
train_data = DictDataset({'t': T_train_Nf, 'y': Y_train_Nf, 'x': X_train_Nf, 'z': Z_train_Nf}, name='train')
# test dataset
test_data = DictDataset({'t': T_test, 'y': Y_test, 'x': X_test, 'z': Z_test,
                         'temperature': temp_test}, name='test')

# torch dataloaders
batch_size = X_train_Nf.shape[0]  # full batch training
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           collate_fn=train_data.collate_fn,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          collate_fn=test_data.collate_fn,
                                          shuffle=False)

from neuromancer.modules import blocks
from neuromancer.system import Node

# neural net to solve the PDE problem bounded in the PDE domain
net = blocks.MLP(insize=4, outsize=1, hsizes=[32, 32], nonlin=nn.LogSigmoid)

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
rho = 2092
C_p = 678
k = 18.2


f_pinn = (rho * C_p * dtemperature_dt - k * d2temperatur_d2x - k * d2temperatur_d2y - k * d2temperatur_d2z
          - total_heat_train_Nf)

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
ell_u = scaling * (temperature_hat[-Nu:] == temp_train_Nu) ^ 2  # remember we stacked CP with IC and BC

# # output constraints to bound the PINN solution in the PDE output domain [-1.0, 1.0]

con_1 = scaling * (temperature_hat <= min_max(temp_test)[1]) ^ 2
con_2 = scaling * (temperature_hat >= min_max(temp_test)[0]) ^ 2

from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem

# create Neuromancer optimization loss
pinn_loss = PenaltyLoss(objectives=[ell_f, ell_u], constraints=[con_1, con_2])

# construct the PINN optimization problem
problem = Problem(nodes=[pde_net],  # list of nodes (neural nets) to be optimized
                  loss=pinn_loss,  # physics-informed loss function
                  grad_inference=True  # argument for allowing computation of gradients at the inference time)
                  )

from neuromancer.trainer import Trainer

optimizer = torch.optim.Adam(problem.parameters(), lr=0.01)
epochs = 5000

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
temperature_pinn = temperature1.reshape(shape=[file_num, 1552]).detach().cpu()

contour(temperature_pinn[89, :], 1)

contour(temperature_pinn[89, :]-temp[89, :], 1)

print("0")
