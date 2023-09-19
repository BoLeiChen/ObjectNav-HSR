import matplotlib.pyplot as plt
import numpy as np
import sys

_path_ours = '/home/cbl/RelationalGraphLearning_origin/crowd_nav/data/output/output.log'
_path_RGL = 'output_RGL.log'
_path_SGD3QN = 'output_SGD3QN.log'
_path_IRN = 'output_IRN.log'
delim =' '

with open(_path_ours, 'r') as f:
	reward_ours = []
	for line in f:
	    line = line.strip().split(delim)
	    #if len(line)>23:
	    #	print(line[22])
	    if len(line)>=26 and line[23]=='return:':
		
	    	#print(line[24])
	    	reward_ours.append(float(line[24][:-1]))
reward_ours = np.asarray(reward_ours)

'''
with open(_path_RGL, 'r') as f:
	reward_RGL = []
	for line in f:
	    line = line.strip().split(delim)
	    if len(line)>=26 and line[24]=='return:':
	    	reward_RGL.append(float(line[25]))
reward_RGL = np.asarray(reward_RGL)

with open(_path_SGD3QN, 'r') as f:
	reward_SGD3QN = []
	for line in f:
	    line = line.strip().split(delim)
	    if len(line)>23 and line[21]=='reward:':
	    	reward_SGD3QN.append(float(line[22].strip().split(',')[0]))
reward_SGD3QN = np.asarray(reward_SGD3QN)

with open(_path_IRN, 'r') as f:
	reward_IRN = []
	for line in f:
	    line = line.strip().split(delim)
	    if len(line)>23 and line[21]=='reward:':
	    	reward_IRN.append(float(line[22].strip().split(',')[0]))
reward_IRN = np.asarray(reward_IRN)
''' 
#epoch,acc,loss,val_acc,val_loss

dis = 20
'''
for i in range(len(reward_ours)-dis):
	for j in range(dis-1):
		reward_ours[i] +=reward_ours[i+j+1]
	reward_ours[i] = reward_ours[i]/dis

for i in range(len(reward_RGL)-dis):
	for j in range(dis-1):
		reward_RGL[i] +=reward_RGL[i+j+1]
	reward_RGL[i] = reward_RGL[i]/dis

for i in range(len(reward_SGD3QN)-dis):
	for j in range(dis-1):
		reward_SGD3QN[i] +=reward_SGD3QN[i+j+1]
	reward_SGD3QN[i] = reward_SGD3QN[i]/dis

for i in range(len(reward_IRN)-dis):
	for j in range(dis-1):
		reward_IRN[i] +=reward_IRN[i+j+1]
	reward_IRN[i] = reward_IRN[i]/dis

'''
y_axis_data_ours = reward_ours[:len(reward_ours)-dis]
#y_axis_data_RGL = reward_RGL[:len(reward_RGL)-dis]
#y_axis_data_SGD3QN = reward_SGD3QN[:len(reward_SGD3QN)-dis]
#y_axis_data_IRN = reward_IRN[:len(reward_IRN)-dis]

x_axis_data_ours = []
for i in range(len(y_axis_data_ours)):
	x_axis_data_ours.append(i)
'''
x_axis_data_RGL = []
for i in range(len(y_axis_data_RGL)):
	x_axis_data_RGL.append(i)
x_axis_data_SGD3QN = []
for i in range(len(y_axis_data_SGD3QN)):
	x_axis_data_SGD3QN.append(i)
x_axis_data_IRN = []
for i in range(len(y_axis_data_IRN)):
	x_axis_data_IRN.append(i)
'''
#y_axis_data3 = [82,83,82,76,84,92,81]

        
#画图 
plt.plot(x_axis_data_ours, y_axis_data_ours, 'b', alpha=0.3, label='ours')#'
#plt.plot(x_axis_data_RGL, y_axis_data_RGL, 'r', alpha=0.01, label='RGL')
#plt.plot(x_axis_data_SGD3QN, y_axis_data_SGD3QN, 'g', alpha=0.5
#, label='SG-D3QN')
#plt.plot(x_axis_data_IRN, y_axis_data_IRN, 'y', alpha=0.3, label='IRN')

#plt.plot(x_axis_data, y_axis_data3, 'go--', alpha=0.5, linewidth=1, label='acc')

 
plt.legend()  #显示上面的label
plt.xlabel('epoch')
plt.ylabel('reward')#accuracy
 
#plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()

