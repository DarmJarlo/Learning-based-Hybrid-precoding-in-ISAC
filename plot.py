import matplotlib.pyplot as plt

# Read data from the first file
file1 = 'loss1.txt'
loss1 = []
with open(file1, 'r') as f1:
    for line in f1:
        loss1.append(float(line.strip()))

# Read data from the second file
file2 = 'loss2.txt'
loss2 = []
with open(file2, 'r') as f2:
    for line in f2:
        loss2.append(float(line.strip()))

# Read data from the third file
file3 = 'loss3.txt'
loss3 = []
with open(file3, 'r') as f3:
    for line in f3:
        loss3.append(float(line.strip()))

# Plotting the loss-iterations graph
iterations = range(1, len(loss1) + 1)

plt.plot(iterations, loss1, label='Loss 1')
plt.plot(iterations, loss2, label='Loss 2')
plt.plot(iterations, loss3, label='Loss 3')

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss-Iterations Graph')
plt.legend()
plt.grid(True)
plt.show()
