import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

KERNEL=3

with open(r'/Users/user/Library/CloudStorage/GoogleDrive-jtupayac@vols.utk.edu/My Drive/Own Reseach/FLResults/multiples/2clients/1684496687/out_server_2.txt3',"r") as fi:
    file1 = []
    for ln in fi:
        if ln.startswith("metrics.matthews_corrcoef => "):
            file1.append(float(ln.split("=> ")[-1].replace("\n","")))
file1=signal.medfilt(file1,KERNEL)
print(file1)


with open(r'/Users/user/Library/CloudStorage/GoogleDrive-jtupayac@vols.utk.edu/My Drive/Own Reseach/FLResults/multiples/2clients/1684506811/out_server_2.txt3',"r") as fi:
    file2 = []
    for ln in fi:
        if ln.startswith("metrics.matthews_corrcoef => "):
            file2.append(float(ln.split("=> ")[-1].replace("\n","")))
file2=signal.medfilt(file2,KERNEL)
print(file2)




with open(r'/Users/user/Library/CloudStorage/GoogleDrive-jtupayac@vols.utk.edu/My Drive/Own Reseach/FLResults/multiples/2clients/1684506903/out_server_2.txt3',"r") as fi:
    file3= []
    for ln in fi:
        if ln.startswith("metrics.matthews_corrcoef => "):
            file3.append(float(ln.split("=> ")[-1].replace("\n","")))
file3=signal.medfilt(file3,KERNEL)
print(file3)



x = file1#[10,20,30,40,50]
y = file2#[30,30,30,30,30]


# plot lines
plt.plot(range(len(file1)),file1, label = "Dataset 687")
plt.plot( range(len(file2)),file2, label = "Dataset 811")
plt.plot( range(len(file3)),file3, label = "Dataset 903")
plt.title("MCC")
plt.xlabel("FL Rounds")
plt.ylabel("Mathew Correlation Coeficient")

plt.legend()
plt.show()


print("#########NEW@##########")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


with open(r'/Users/user/Library/CloudStorage/GoogleDrive-jtupayac@vols.utk.edu/My Drive/Own Reseach/FLResults/multiples/2clients/1684496687/out_server_2.txt3',"r") as fi:
    file1 = []
    for ln in fi:
        if ln.startswith("   macro avg"):
            file1.append(float(ln.split("     ")[-2]))
file1=signal.medfilt(file1,KERNEL)
print(file1)



with open(r'/Users/user/Library/CloudStorage/GoogleDrive-jtupayac@vols.utk.edu/My Drive/Own Reseach/FLResults/multiples/2clients/1684506811/out_server_2.txt3',"r") as fi:
    file2 = []
    for ln in fi:
        if ln.startswith("   macro avg"):
            file2.append(float(ln.split("     ")[-2]))
file2=signal.medfilt(file2,KERNEL)
print(file2)




with open(r'/Users/user/Library/CloudStorage/GoogleDrive-jtupayac@vols.utk.edu/My Drive/Own Reseach/FLResults/multiples/2clients/1684506903/out_server_2.txt3',"r") as fi:
    file3= []
    for ln in fi:
        if ln.startswith("   macro avg"):
            file3.append(float(ln.split("     ")[-2]))
file3=signal.medfilt(file3,KERNEL)
print(file3)



x = file1#[10,20,30,40,50]
y = file2#[30,30,30,30,30]

# plot lines
plt.plot(range(len(file1)),file1, label = "Dataset 687")
plt.plot( range(len(file2)),file2, label = "Dataset 811")
plt.plot( range(len(file3)),file3, label = "Dataset 903")
plt.title("F1 - Macro Score")
plt.xlabel("FL Rounds")
plt.ylabel("F1 - Macro Score")

plt.legend()
plt.show()


