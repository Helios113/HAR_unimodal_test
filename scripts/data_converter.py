import numpy as np
import pandas as pd
from pathlib import Path
import time

dir = Path("data")
data_ext=".log"

t1s = time.time();
li = None
# Get all files in the data directory with
# appropriate extension
for file in dir.glob('**/*'+data_ext):
    # read each file
    data = pd.read_csv(file, delimiter='\t', header=None, dtype=np.float64).values
    # eliminate null data
    # and create a big list in memory
    if li is None:
        li = data[data[:,-1]!=0,:]
    else:
        li = np.append(li,data[data[:,-1]!=0,:], axis=0)
# separate data from labels
labels = li[:, 23:].astype(int)
# data = (li[:, :23]-np.min(li[:, :23],axis=0))/(np.max(li[:, :23],axis=0)-np.min(li[:, :23],axis=0))
data = li[:, :23]
t1e = time.time();
# Define the window size and the stride
# Window size
WINDOW_SIZE = 20
# Stride
STRIDE = 2
t2s=time.time();
windows = None
window_labels = None
spinning_cursor = '|/-\\'
k=0
# iterate through windows
for i in range(0, data.shape[0]-WINDOW_SIZE, STRIDE):
    # create data structure with data and labels

    # logger
    print("",end='\r')
    print(r'Working: {per:6.2f}%'.format(per=100*(i+1)/(data.shape[0]-WINDOW_SIZE)),r'{cur}'.format(cur=spinning_cursor[int(t2s-time.time())%4]), end='')

    # Data stacking
    if windows is None:
        windows = data[i:i+WINDOW_SIZE, 0:3].reshape(1,-1)
    else:
        windows = np.vstack((windows,data[i:i+WINDOW_SIZE, 0:3].reshape(1,-1)))
    
    # Label stacking
    l = np.zeros(13)
    for j in labels[i:i+WINDOW_SIZE]:
        l[j]+=1
    if window_labels is None:
        window_labels = l.argmax()
    else:
        window_labels = np.vstack((window_labels,l.argmax()))
print("\n")
t2e = time.time();
print(r"Data loading took: {tm:>21.5f}s".format(tm=t1e-t1s))
print(r"Data partitioning took: {tm:>16.5f}s".format(tm=t2e-t2s))
# save data
np.savetxt(dir/"data.csv", windows, delimiter=",")
np.savetxt(dir/"labels.csv", window_labels, delimiter=",")
# there is an empty line at the end of the csv