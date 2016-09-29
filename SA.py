import numpy as np
import random
import matplotlib.pyplot as plt
n = 4

count = 1
J = np.zeros(shape = (n,n))
for i in range(n):
	for j in range (n):
		J[i][j] = count
		count = count + 1

init_1 = [1,1]
init_2 = [2,2]

minloc = [0,0]
minval = 100

print "The function map is: "
print J

def neighbor(loc):
	if (loc[0]==0) & (loc[1]==0):
		m = random.randint(0,1) # down, right
		if m == 0:
			return [1,0]
		else:
			return [0,1]
	if (loc[0]==0) & (loc[1]==n-1):
		m = random.randint(0,1) # down, left
		if m == 0:
			return [1,n-1]
		else:
			return [0,n-2]
	if (loc[0]==n-1) & (loc[1]==0):
		m = random.randint(0,1) # up, right
		if m == 0:
			return [n-2,0]
		else:
			return [n-1,1]
	if (loc[0]==n-1) & (loc[1]==n-1):
		m = random.randint(0,1) # up, left
		if m == 0:
			return [n-2,n-1]
		else:
			return [n-1,n-2]
	if (loc[0]==0):
		m = random.randint(0,2) # down, left, right
		if m == 0:
			return [1,loc[1]]
		if m == 1:
			return [0,loc[1]-1]
		if m == 2:
			return [0,loc[1]+1]
	if (loc[0]==n-1):
		m = random.randint(0,2) # up, left, right
		if m == 0:
			return [n-2,loc[1]]
		if m == 1:
			return [n-1,loc[1]-1]
		if m == 2:
			return [n-1,loc[1]+1]
	if (loc[1]==0):
		m = random.randint(0,2) # up, down, right
		if m == 0:
			return [loc[0]-1,loc[1]]
		if m == 1:
			return [loc[0]+1,loc[1]]
		if m == 2:
			return [loc[0],loc[1]+1]
	if (loc[1]==n-1):
		m = random.randint(0,2) # up, down, left
		if m == 0:
			return [loc[0]-1,loc[1]]
		if m == 1:
			return [loc[0]+1,loc[1]]
		if m == 2:
			return [loc[0],loc[1]-1]
	m = random.randint(0,3) # left, right, up, down
	if m == 0:
		return [loc[0],loc[1]-1]
	if m == 1:
		return [loc[0],loc[1]+1]
	if m == 2:
		return [loc[0]-1,loc[1]]
	if m == 3:
		return [loc[0]+1,loc[1]]

map = np.zeros(shape = (n,n))
Temperature = 5000
temp = []
for i in range(10):

	if J[init_1[0]][init_1[1]] < minval:
		minval = J[init_1[0]][init_1[1]]
		minloc = init_1
	if J[init_2[0]][init_2[1]] < minval:
		minval = J[init_2[0]][init_2[1]]
		minloc = init_2

	map[init_1[0]][init_1[1]] = J[init_1[0]][init_1[1]]
	map[init_2[0]][init_2[1]] = J[init_2[0]][init_2[1]]

	new_1 = neighbor(init_1)
	if map[new_1[0]][new_1[1]] != 0:
		new_1 = neighbor(init_1)
	if map[new_1[0]][new_1[1]] != 0:
		new_1 = neighbor(init_1)
	if map[new_1[0]][new_1[1]] != 0:
		new_1 = neighbor(init_1)

	new_2 = neighbor(init_2)
	if map[new_2[0]][new_2[1]] != 0:
		new_2 = neighbor(init_2)
	if map[new_2[0]][new_2[1]] != 0:
		new_2 = neighbor(init_2)
	if map[new_2[0]][new_2[1]] != 0:
		new_2 = neighbor(init_2)

	if J[new_1[0]][new_1[1]] <= J[init_1[0]][init_1[1]]:
		init_1 = new_1
	else:
		if random.random() < np.exp(-(J[new_1[0]][new_1[1]] - J[init_1[0]][init_1[1]])/Temperature):
			init_1 = new_1

	if J[new_2[0]][new_2[1]] <= J[init_2[0]][init_2[1]]:
		init_2 = new_2
	else:
		if random.random() < np.exp(-(J[new_2[0]][new_2[1]] - J[init_2[0]][init_2[1]])/Temperature):
			init_2 = new_2

	if Temperature < 800:
		Temperature = Temperature * 0.5
	else:
		Temperature = Temperature * 0.3

	temp.append(Temperature)

	print minval, "now at", minloc

x = np.arange(0,len(temp),1)
plt.plot(x,temp)
plt.show()
