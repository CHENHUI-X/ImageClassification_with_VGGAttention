from PIL import Image
import pickle
import os
import numpy as np
import random
import torch
import cv2
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder
ISCUDA = torch.cuda.is_available()
import torch.nn.functional as F
import time
from tqdm import tqdm


# -----------------------------用于解压batch文件-----------------------------
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# -----------------------------用于获取batch中的数据-----------------------------

def getimageinfo(folder_path):
    # get dirpath , subdirpath , filename

    ''''
    The data information:
        http://www.cs.toronto.edu/~kriz/cifar.html
        batches.meta. It too contains a Python dictionary object.
        It has the following entries:
        label_names -- a 10-element list which gives
        meaningful names to the numeric labels in the labels
        array described above. For example,
        label_names[0] == "airplane", label_names[1] == "automobile", etc.

        print(images[0].keys())

        dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
        **
            for every batch file
            data -- a 10000x3072 numpy array of uint8s.
            Each row of the array stores a 32x32 colour image.
            The first 1024 entries contain the red channel values,
            the next 1024 the green, and the final 1024 the blue.
            The image is stored in row-major order,
            so that the first 32 entries of the array are the red channel
            values of the first row of the image.

        **
            labels -- a list of 10000 numbers in the range 0-9.
            The number at index i indicates the label of the ith
            image in the array data.

    '''
    filelist = list(os.walk(folder_path))[0]

    # get batch file
    batchlist = [filelist[0] + '/'+i for i in filelist[-1] if '_' in i]

    # get data
    data = [ unpickle(i) for i  in batchlist  ]

    return  data

# -----------------------------用于获取batch中data和label的数据-----------------------------
def getdata(imagedatalist : list) -> any:
    encode = OneHotEncoder()
    batch_merge_data = []
    batch_merge_label = []
    for image_dic_i in range(len(imagedatalist)):
        batch_i_items = imagedatalist[image_dic_i]
        batch_i_data = np.array(batch_i_items[b'data'])
        # print(batch1data.shape) # (10000, 3072)
        batch_i_labels = np.array(
            batch_i_items[b'labels']).reshape((-1,1)) # (10000, 1)
        # transform to onehot encode
        batch_i_labels = encode.fit_transform(
            batch_i_labels).toarray()  # (10000, 10)

        batch_merge_data.append(batch_i_data)
        batch_merge_label.append(batch_i_labels)

    Data = np.array(batch_merge_data).reshape(-1,3072) # (60000, 3072)
    Label = np.array(batch_merge_label).reshape(-1,10) # (60000, 10)


    return Data[:-10000],Label[:-10000],Data[-10000:],Label[-10000:]

# -----------------------------调用数据提取函数-----------------------------
'''
Note : 
    if you are using the CIFAR10 dataset for the first time, 
    you can download the data using the code below:
    
        from torchvision import datasets
        dataset = datasets.CIFAR10(
            root='./data', 
            download=True,
        )
        images_info_list = getimageinfo('./data/cifar-10-batches-py')
        # 5 batch for train & val , 1 batch for test

'''

images_info_list = getimageinfo('./data/cifar-10-batches-py')
data,labels,test_data,test_labels = \
    getdata(imagedatalist=images_info_list)

'''
print(data.shape,labels.shape,test_data.shape,test_labels.shape)
(50000, 3072) (50000, 10) (10000, 3072) (10000, 10)
'''

# ---------------------- process data and transform to tensor-------------------

class ImageDataset(Dataset):

    def __init__(self, data , labels, transform = transforms.ToTensor(), mode='train'):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = np.array(self.data[idx])
        # print(img.shape) # 3072
        R = img[:1024].reshape(32, 32)
        G = img[1024:1024 * 2].reshape(32, 32)
        B = img[1024 * 2:].reshape(32, 32)

        Images = np.array(cv2.merge((R, G, B)))
        # 注意如果pytorch卷积的话,需要channel数目在前.
        # print(Images.shape) # ( 32, 32, 3 )

        img = Image.fromarray(Images,'RGB') # need (H,W,C) -> (H,W,C) image
        # print(np.shape(img)) # (32, 32, 3)

        img = self.transform(img)  # images agreemention
        # print(np.shape(img)) # now the  shape is (3, 32, 32) then resize to ( 3 , 224,224)
        '''
        Note 1 :
            Here transform input need a img , because the 
            transforms.RandomRotation(10)
            transforms.RandomHorizontalFlip()
            just used for  images, so  we  need  transform array to  image ,
            then call the transform function .
        Note 2 : 
            ToTensor() Converts a PIL Image or numpy.ndarray (H x W x C) 
            in the range [0, 255] to a torch.FloatTensor 
            of shape (C x H x W) in the range [0.0, 1.0] 
            
            then you will see that  img shape has  changed.
            (32, 32, 3) -> (3, 32, 32) , surely we resize the image,
            sou you actually see that shape should be (3, 224,224)
        
        '''

        label = self.labels[idx]
        if ISCUDA:
            img = img.cuda()
            label = label.cuda()

        return img, label


# --------------Define the attention block --------------

class AttentionBlock(nn.Module):
    '''
     we make the  output of pool-3 or pool-4 to infer attention
     that is pool-3 pass through in a CNN  to change channel ,
     at same time , put the last pool output(poo-5) in a CNN layer
     to change channel , then add the 3 & 5 (Same goes for pool-4),
     put it in relu to get a attention map

    '''
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=True):
        super(AttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)

    def forward(self, l, g):
        N, C, W, H = l.shape
        l_ = self.W_l(l) # pool-3 or pool-4 output
        g_ = self.W_g(g) # the global feature  (pool-5 output)
        if self.up_factor > 1:
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(F.relu(l_ + g_))  # batch_size x 1 x W x H - attention map

        # compute attn weight map
        if self.normalize_attn:
            #   flatten attention map and make it normalization
            #  attention weight map elements sum is 1
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H) # transform to weight
        else:
            # just make attention map elements to [0,1]
            a = torch.sigmoid(c)
        '''
            Now a is normalized Attention map,Each scalar element 
            in a represents the degree of attention to the corresponding 
            spatial feature vector in pool-3 ( or pool-4).
        '''


        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l)
        # a.expand_as(l) : repeat the a alone channel
        # batch_size x C x W x H elements multiplication  -> get attention result : batch_size x C x W x H

        if self.normalize_attn:
            output = f.view(N, C, -1).sum(dim=2)
            # weighted sum alone channel ,that is sum of h and w dim
        else:
            output = F.adaptive_avg_pool2d(f, (1, 1)).view(N, C)  # global average pooling

        return a, output
        # where a is the attention map (have one channel )
        #  output is  a  vector with shape (N,C) , then connect with others attention result


class VGG16Att(nn.Module):
    def __init__(self, num_classes = None, normalize_attn = False, dropout= None):
        super(VGG16Att, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # pool-1 -> size / 2 : 112
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # pool-2 -> size / 2 : 56
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # pool-3 -> size / 2 : 28
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # pool-4 -> size / 2 : 14
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # pool-5 -> size / 2 : 7

        '''

            here we use attention feature , do not use VGG original fc layers:
                    self.fc = nn.Sequential(
                        nn.Dropout(0.5),
                        # nn.Linear(7 * 7 * 512, 4096),
                        nn.Linear(1*1*512,4096),
                        nn.ReLU())

                    self.fc1 = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(4096, 4096),
                        nn.ReLU())
                    self.fc2 = nn.Sequential(
                        nn.Linear(4096, num_classes))
        '''

        self.pool = nn.AvgPool2d(7, stride=1)
        # initialize the attention blocks defined above
        # parameter is  in_features_l, in_features_g, attn_features, up_factor
        self.attn1 = AttentionBlock(256, 512, 256, 4, normalize_attn=normalize_attn)
        self.attn2 = AttentionBlock(512, 512, 256, 2, normalize_attn=normalize_attn)
        self.dpt = None
        if dropout is not None:
            self.dpt = nn.Dropout(dropout)
        self.cls = nn.Linear(in_features = 512 + 512 + 256, out_features = num_classes, bias=True)
        self.sigmiod = nn.Sigmoid()
        # attention pool-3 + attention pool-4 , AvgPool2d pool-5 : = 512 + 512 + 256


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        pool3 = self.layer7(out) # pool-3
        out = self.layer8(pool3)
        out = self.layer9(out)
        pool4 = self.layer10(out) # pool-4
        out = self.layer11(pool4)
        out = self.layer12(out)
        pool5 = self.layer13(out) # pool-5

        # this code is original , we do not use this
        # out = out.reshape(out.size(0), -1)
        # out = self.fc(out)
        # out = self.fc1(out)
        # out = self.fc2(out)
        N, __, __, __ = pool5.shape

        g = self.pool(pool5).view(N, 512)
        a1, g1 = self.attn1(pool3, pool5)
        a2, g2 = self.attn2(pool4, pool5)
        # a1，a2 is the  attention map for pool-3 and pool-5
        # g1，g2 is
        g_hat = torch.cat((g, g1, g2), dim=1)  # batch_size x C
        if self.dpt is not None: # drop put
            g_hat = self.dpt(g_hat)
        out = self.cls(g_hat)
        out = self.sigmiod(out) # pro
        return [out, a1, a2]

# --------------define parameter , model , loss function ed --------------
validation_size = 0.2
batchsize = 1
numclass = 10
dropout = 0.1
learning_rate = 0.01
num_epochs = 2
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VGG16Att(num_classes = numclass, normalize_attn=True,dropout = dropout)
if ISCUDA:
    model= model.cuda()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)


# --------------Split train into train and val,then get it in batch --------------

X_train, X_val, y_train, y_val = train_test_split(
    data,labels, test_size = validation_size, random_state=0
)

train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]) ]
        )

val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

train_dataset = ImageDataset(
    data = X_train,labels = y_train,transform = train_transform)
val_dataset = ImageDataset(
    data = X_val,labels  = y_val,transform = val_transform)
test_dataset = ImageDataset(
    data = test_data,labels  = test_labels,transform = val_transform)

train_loader = DataLoader(train_dataset,batch_size= batchsize,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size= batchsize,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size= batchsize,shuffle=False)

# --------------Start train  and  optimization  --------------

start_time = time.time()
train_losses = []
val_losses = []
step = 0
for epoch in range(num_epochs):

    train_preds = []
    train_targets = []
    auc_train = []
    loss_epoch_train_for_batch = []
    loss_epoch_val_for_batch = []
    # Run the training batches
    for batch_index, (X_train, y_train) in tqdm(enumerate(train_loader), total=len(train_loader)):
        step += 1
        y_pred, _, _ = model(X_train)
        loss = criterion(
            y_pred.type(torch.FloatTensor), y_train.type(torch.FloatTensor))
        loss_epoch_train_for_batch.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch [{}/{}] - step {} - Loss: {:.4f}'
              .format(epoch + 1, num_epochs, step, loss.item()))

    train_losses.append(np.mean(loss_epoch_train_for_batch))

    print(f'Epoch [{epoch:2}/{num_epochs:2}] ----- Average_loss: {np.mean(loss_epoch_train_for_batch):10.8f } ')

    # Run the validate  batches
    with torch.no_grad():
        for b, (X_val, y_val) in enumerate(val_loader):
            y_val, _, _ = model(X_val)
            loss = criterion(torch.sigmoid(y_val.type(torch.FloatTensor)), y_val.type(torch.FloatTensor))
            loss_epoch_val_for_batch.append(loss.item())

    val_losses.append(np.mean(loss_epoch_val_for_batch))
    print(f'\nEpoch: {epoch} Val Loss: {np.mean(loss_epoch_val_for_batch):10.8f} ')

print(f'\nTime spent : {time.time() - start_time:.0f} seconds')  # print the time elapsed


















