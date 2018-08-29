
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


# Create images with random rectangles and bounding boxes. 
num_imgs = 50000

img_size = 8
min_object_size = 1
max_object_size = 4
num_objects = 1

bboxes = np.zeros((num_imgs, num_objects, 4))
imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0

for i_img in range(num_imgs):
    for i_object in range(num_objects):
        w, h = np.random.randint(min_object_size, max_object_size, size=2)
        x = np.random.randint(0, img_size - w)
        y = np.random.randint(0, img_size - h)
        imgs[i_img, x:x+w, y:y+h] = 1.  # set rectangle to 1
        bboxes[i_img, i_object] = [x, y, w, h]
        
imgs.shape, bboxes.shape


# In[3]:


i = 0
plt.imshow(imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
for bbox in bboxes[i]:
    plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))


# In[4]:


# Reshape and normalize the image data to mean 0 and std 1. 
X = torch.from_numpy((imgs.astype(np.float32).reshape(num_imgs, -1) - np.mean(imgs)) / np.std(imgs))


# In[5]:


# Normalize x, y, w, h by img_size, so that all values are between 0 and 1.
# Important: Do not shift to negative values (e.g. by setting to mean 0), because the IOU calculation needs positive w and h.
y = torch.from_numpy(bboxes.astype(np.float32).reshape(num_imgs, -1) / img_size)


# In[6]:

# Split training and test.
i = int(0.8 * num_imgs)
train_X = X[:i]
test_X = X[i:]
train_y = y[:i]
test_y = y[i:]
test_imgs = imgs[i:]
test_bboxes = bboxes[i:]

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

optimizer = optim.SGD(net.parameters(), lr=1)

epochs = 1
for epoch in range(epochs):

    for mini_batch in batches:
        optimizer.zero_grad()
        
        output = net(mini_batch[0])
    
        criterion = nn.MSELoss()
        loss = criterion(output, mini_batch[1])

        loss.backward()

        optimizer.step()

    print(epoch, '-', float(loss))

'''
output = net(test_X)
test_X = test_X.view(-1, 8, 8).numpy()
output = output.detach().numpy() * img_size

for i in range(8):
    plt.imshow(test_X[i], origin='upper', extent=[0, img_size, img_size, 0])
    plt.gca().add_patch(matplotlib.patches.Rectangle((output[i][0], output[i][1]), output[i][2], output[i][3], ec='r', fc='none'))
    plt.show()
'''

'''
# In[7]:


# Build the model.

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
model = Sequential([
        Dense(200, input_dim=X.shape[-1]), 
        Activation('relu'), 
        Dropout(0.2), 
        Dense(y.shape[-1])
    ])
model.compile('adadelta', 'mse')

# In[8]:


# Train.
model.fit(train_X, train_y, nb_epoch=30, validation_data=(test_X, test_y), verbose=2)

# In[9]:


# Predict bounding boxes on the test images.
pred_y = model.predict(test_X)
pred_bboxes = pred_y * img_size
pred_bboxes = pred_bboxes.reshape(len(pred_bboxes), num_objects, -1)
pred_bboxes.shape
'''
pred_y = net(test_X)
pred_bboxes = pred_y.detach().numpy() * img_size
pred_bboxes = pred_bboxes.reshape(len(pred_bboxes), num_objects, -1)
pred_bboxes.shape

# In[10]:


def IOU(bbox1, bbox2):
    #Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0.
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U


# In[22]:


# Show a few images and predicted bounding boxes from the test dataset. 
plt.figure(figsize=(12, 3))
for i_subplot in range(1, 5):
    plt.subplot(1, 4, i_subplot)
    i = np.random.randint(len(test_imgs))
    plt.imshow(test_imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
    for pred_bbox, exp_bbox in zip(pred_bboxes[i], test_bboxes[i]):
        plt.gca().add_patch(matplotlib.patches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3], ec='r', fc='none'))
        plt.annotate('IOU: {:.2f}'.format(IOU(pred_bbox, exp_bbox)), (pred_bbox[0], pred_bbox[1]+pred_bbox[3]+0.2), color='r')
        
# plt.savefig('plots/bw-single-rectangle_prediction.png', dpi=300)


# In[23]:


# Calculate the mean IOU (overlap) between the predicted and expected bounding boxes on the test dataset. 
summed_IOU = 0.
for pred_bbox, test_bbox in zip(pred_bboxes.reshape(-1, 4), test_bboxes.reshape(-1, 4)):
    summed_IOU += IOU(pred_bbox, test_bbox)
mean_IOU = summed_IOU / len(pred_bboxes)
mean_IOU

plt.show()
