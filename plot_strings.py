import glob
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
p = natsorted(glob.glob('./string_out/centers_iter*.dat'))
for data in p:
    dataa = np.genfromtxt(data)[:, 1:3]*10
    plt.plot(dataa[:, 0], dataa[:, 1])
plt.savefig('./strings_old.png')
plt.close()
for it in range(1, 5):
    linelist = []
    for img in range(1, 51):
        try:
            with open(f'./string_out/iter{it}/img{img}/equi20/6VJM_out_iter{it}_img{img}_equi20_out.colvars.traj', 'r') as f:
                lines = [float(j) for j in f.readlines()[-1].split()]
        except FileNotFoundError:
            continue
        if lines[0] == 1000:
            linelist.append(lines)
            print(lines)
    linelist = np.array(linelist)
    if linelist.ndim == 2:
        plt.plot(linelist[:, 4], linelist[:, 5])
plt.savefig('./strings_equil.png')
    #print(linelist.shape)
    #print(dataa.shape)
