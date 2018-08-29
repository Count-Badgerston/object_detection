import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

num_imgs = 50000

img_size = 8
min_object_size = 1
max_object_size = 4
num_objects = 1

imgs = np.zeros((num_imgs, img_size, img_size))
boxes = np.zeros((num_imgs, 4))

for img in range(num_imgs):

    w, h = np.random.randint(min_object_size, max_object_size, size=2)
    x = np.random.randint(0, img_size-w)
    y = np.random.randint(0, img_size-h)
    imgs[img, y:y+h, x:x+w] = 1

    boxes[img] = [x, y, w, h]

#X = torch.from_numpy(imgs.astype(np.float32).reshape(-1, 1, 8, 8) * 0.1)
X = torch.from_numpy(imgs.astype(np.float32).reshape(-1, 64) * 0.1)
y = torch.from_numpy(boxes.astype(np.float32) * 0.1)

i = int(0.8 * num_imgs)
train_X = X[:i]
train_y = y[:i]
test_X = X[i:]
test_y = y[i:]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(1, 8, 2, padding=1)
        #self.conv2 = nn.Conv2d(8, 8, 2, padding=1)
        self.fc1 = nn.Linear(train_X.size(1), 200)
        self.fc2 = nn.Linear(200, train_y.size(1))
        #self.drop_out = nn.Dropout(p=0.3)

    def forward(self, x):
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        #x = x.view(x.size(0), -1)
        #x = self.drop_out(x)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

net = Net()

def prepare_mini_batches(train_set, target_set):
    if train_set.size(0) != target_set.size(0):
        print("train and target sets don't match !")
        exit()
    mini_batch_size = 1
    batches = []
    for m in range(0, train_set.size(0), mini_batch_size):
        mini_batch = (train_set[m:m+mini_batch_size], target_set[m:m+mini_batch_size])
        batches.append(mini_batch)
    return batches

batches = prepare_mini_batches(train_X, train_y)

optimizer = optim.SGD(net.parameters(), lr=1)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [20], gamma=0.8)

epochs = 50
for epoch in range(epochs):

    #scheduler.step()

    for mini_batch in batches:
        optimizer.zero_grad()
        
        output = net(mini_batch[0])
    
        criterion = nn.MSELoss()
        loss = criterion(output, mini_batch[1])

        loss.backward()

        optimizer.step()

    print(epoch, '-', float(loss))

output = net(test_X)
test_X = test_X.view(-1, 8, 8).numpy() * 10
output = output.detach().numpy() * 10

for i in range(8):
    plt.imshow(test_X[i], origin='upper', extent=[0, img_size, img_size, 0])
    plt.gca().add_patch(matplotlib.patches.Rectangle((output[i][0], output[i][1]), output[i][2], output[i][3], ec='r', fc='none'))
    plt.show()


'''
bboxes = np.zeros((num_imgs, num_objects, 4))
imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0

for i_img in range(num_imgs):
    for i_object in range(num_objects):
        w, h = np.random.randint(min_object_size, max_object_size, size=2)
        x = np.random.randint(0, img_size - w)
        y = np.random.randint(0, img_size - h)
        imgs[i_img, x:x+w, y:y+h] = 1.  # set rectangle to 1
        bboxes[i_img, i_object] = [x, y, w, h]
        print(x, y, w, h)

print(imgs[0])
plt.imshow(imgs[0], origin='lower')
plt.gca().add_patch(matplotlib.patches.Rectangle((x-0.5, y-0.5), w, h, ec='r', fc='none'))
plt.show()

plt.imshow(imgs[0].T, origin='lower', extent=[0, img_size, 0, img_size])
for bbox in bboxes[0]:
    plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))
plt.show()

i = 0
#plt.imshow(imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
#

X = (imgs.reshape(num_imgs, -1) - np.mean(imgs)) / np.std(imgs)
X.shape, np.mean(X), np.std(X)

y = bboxes.reshape(num_imgs, -1) / img_size
y.shape, np.mean(y), np.std(y)

i = int(0.8 * num_imgs)
train_X = X[:i]
test_X = X[i:]
train_y = y[:i]
test_y = y[i:]
test_imgs = imgs[i:]
test_bboxes = bboxes[i:]



net = Net()

#print(X.shape)
train_input = torch.from_numpy(train_X).float()
train_target = torch.from_numpy(train_y).float()

test_input = torch.from_numpy(test_X).float()
test_target = torch.from_numpy(test_y).float()
#print(input.size())

def prepare_mini_batches(train_set, target_set):
    if train_set.size(0) != target_set.size(0):
        print("train and target sets don't match !")
        exit()
    mini_batch_size = 10
    batches = []
    for m in range(0, train_set.size(0), mini_batch_size):
        mini_batch = (train_set[m:m+mini_batch_size], target_set[m:m+mini_batch_size])
        batches.append(mini_batch)
    return batches
 
data_set = prepare_mini_batches(train_input, train_target)
        

'''