#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys
import time 
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


if torch.cuda.is_available():
    print("cuda is available")
else:
    sys.exit("Cannot run the program because cuda is not available")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        coeffs = torch.rand((W_gpu.shape[1],1))
        self.coeff = nn.Parameter(coeffs) # This intrisically has requires_grad = True !
        
    def forward(self, X):
        return torch.matmul(X, self.coeff)

ip2d = utils.Interp_Wlt_2D()

file = open('file.dat','r')

x = np.array([])
y = np.array([])
f = np.array([])

for line in file:
    line = line.strip()
    if len(line)!=0:
        line_list = line.split()
        values = [float(i) for i in line_list]
        values = np.array(values)
        x = np.append(x, values[0])
        y = np.append(y, values[1]) 
        f = np.append(f, values[2]) 
        
file.close()

x = np.unique(x) ; y = np.unique(y)
xxb, yyb = np.meshgrid(x,y)
zzb = f.reshape(xxb.shape)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(xxb,yyb,zzb, cmap='seismic')
plt.title("Input Surface")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig("./input_surface.jpg")

zz_dxb = ip2d.diff_dx(xxb,yyb,zzb)
zz_dyb = ip2d.diff_dy(xxb,yyb,zzb)

# At times, we need to trim the surface at the boundaries due to presence of abnormality
# If no such abnormality is present, ignore the following 2 lines
xx = xxb[2:99,2:99] ; yy = yyb[2:99,2:99] ; zz = zzb[2:99,2:99] 
zz_dx = zz_dxb[2:99,2:99] ; zz_dy = zz_dyb[2:99,2:99]

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(xx,yy,zz_dy, cmap='copper')
plt.savefig("./input_df_dy.jpg")
ax.plot_surface(xx,yy,zz_dx, cmap='copper')
plt.savefig("./input_df_dx.jpg")

# Builidng the Basis Function Matrix
W_dx = utils.get_BFM(xx,yy,min_j=-2,max_j=1,func=[ip2d.get_scl_derivative,ip2d.get_wlt_derivative],partial='dx')
W_dy = utils.get_BFM(xx,yy,min_j=-2,max_j=1,func=[ip2d.get_scl_derivative,ip2d.get_wlt_derivative],partial='dy')


# Preparation of input for Adam optimization algorithm using PyTorch
W = np.vstack([W_dx,W_dy])
W_gpu = torch.tensor(W).float().cuda()
f_gpu = torch.tensor(np.vstack([zz_dx,zz_dy]).flatten()).float().cuda()

# Starting the training
epochs = 100000
loss_arr = np.array([])

m = Model()

optimizer = optim.Adam(m.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.9,patience=100,verbose=True)

time_s = time.time()
if torch.cuda.is_available():
    m.to(torch.device("cuda:0"))
    print('Running on GPU')
    
    m.train()

    for epoch in range(epochs):
        m.zero_grad()
        output = m.forward(W_gpu)
        loss = F.huber_loss(output.view(f_gpu.shape),f_gpu)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        print(f'epoch: {epoch+1} ; Loss = {loss}')
        loss_arr = np.append(loss_arr, loss.clone().detach().cpu().numpy())

time_e = time.time()

print(f'Time = {time_e-time_s} s')

f_adam = m.forward(W_gpu).clone().detach().cpu().numpy().reshape(f_gpu.shape)
f_adamx = f_adam[0:int(f_adam.size/2)].reshape(zz.shape)
f_adamy = f_adam[int(f_adam.size/2):].reshape(zz.shape) 

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121,projection='3d')
ax2 = fig.add_subplot(122,projection='3d')
ax1.plot_surface(xx,yy,f_adamx,cmap='seismic')
ax2.plot_surface(xx,yy,f_adamy,cmap='seismic')
ax1.set_xlabel('X') ; ax2.set_xlabel('X')
ax1.set_ylabel('Y') ; ax2.set_ylabel('Y')
ax1.set_zlabel('Z') ; ax2.set_zlabel('Z')
plt.savefig("./Reconstructed_derivative_surface.jpg")

params = []

for param in m.parameters():
    params.append(param.data)

coefficients = params[0].clone().detach().cpu().numpy()

fig = plt.figure(figsize=(7,5))
plt.plot(coefficients.flatten())
plt.title('Contribution of Basis functions')
plt.ylabel('Basis function expansion coefficient')
plt.xlabel('Basis functions')
plt.savefig("./func_vs_coeff")

W_mp = utils.get_BFM(xx,yy,min_j=-2,max_j=1,func=[ip2d.get_scl,ip2d.get_wlt])

surf = np.matmul(W_mp,coefficients).reshape(zz.shape)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(xx,yy,surf,cmap='seismic')
ax.set_xlabel('X') 
ax.set_ylabel('Y') 
ax.set_zlabel('Z') 
ax.set_title('Reconstructed Surface')
plt.savefig("./reconstructed_surface.jpg")

