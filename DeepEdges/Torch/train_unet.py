import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
import glob
from scipy import misc
import cv2
import numpy as np
from torchsummary import summary
import torch
import torch.nn as nn
from Torch import pytorch_unet
from Torch import loss
import torch
import torch.optim as optim
import time
import pickle
import imageio as io
#from tensorboardX import SummaryWriter

#writer = SummaryWriter('runs/unet')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = pytorch_unet.UNet(1)
model = model.to(device)

summary(model, input_size=(3, 128, 128))

np.random.seed(17)
torch.manual_seed(17)
# training parameters
batch_size = 32
train_epoch = 100


cuda = True if torch.cuda.is_available() else False

def normalize(I):
    I = I.astype('float32')
    I -= I.mean()
    I /= I.std()
    return I

class EdgesDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir
        files1 = glob.glob(root_dir + '/*.png')
        files2 = glob.glob(root_dir + '/*.gif')
        self.files = files1 + files2
        self.clip = lambda x: np.minimum(np.maximum(x, 0), 1)
        self.sz = (128, 128)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.root_dir,self.files[idx])
        I = io.imread(self.files[idx])
        if len(I.shape) == 3:
            I = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
        I = cv2.resize(I,self.sz)
        B = cv2.Canny(I,0.2,0.7)
        B = B.astype('float32') / 255
        if np.random.randint(2):
            curI = I
        else:
            curI = 255 - I
        curI = curI.astype('float32') / 255
        snr = np.arange(1, 2.1, 0.2)
        s = snr[np.random.randint(len(snr))]
        Y = self.clip(0.1 * (s * curI + np.random.randn(curI.shape[0], curI.shape[1])) + 0.5)
        Y = np.reshape(Y,(1,self.sz[0],self.sz[1]))
        Y = np.concatenate((Y,Y,Y),0)
        B = np.reshape(B,(1,self.sz[0],self.sz[1]))
        Y = torch.tensor(normalize(Y))
        B = torch.tensor(B)
        #sample = {'image': Y, 'edges': B}
        #return sample
        return Y,B

dataset = EdgesDataset('../Dataset/ImageProcessingPlace')
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
image_datasets = {
    'train': train_dataset, 'test': test_dataset
}

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

print(dataset_sizes)


def reverse_transform(I):
    I = I.numpy().transpose((1, 2, 0))
    I-=I.min()
    I/=I.max()
    I = np.clip(I, 0, 1)
    I = (I * 255).astype(np.uint8)
    return I

if False:
    inputs, masks = next(iter(dataloaders['train']))
    print(inputs.shape, masks.shape)
    for x in [inputs.numpy(), masks.numpy()]:
        print(x.min(), x.max(), x.mean(), x.std())
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(reverse_transform(inputs[3]))
    plt.subplot(1,2,2)
    B = reverse_transform(masks[3])
    B = np.reshape(B,(B.shape[0],B.shape[1]))
    plt.imshow(B)
    plt.show()


def calc_loss(pred, target):
    dice = loss.dice_loss(pred, target)
    return dice


def train_model(model, optimizer, num_epochs):
    trainLoss = []
    testLoss = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            epoch_samples = 0
            totalLoss = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels)
                    print('Epoch {}/{}'.format(epoch, num_epochs - 1)+' Loss %2.2f' % loss.item())
                    totalLoss += loss.item()
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            epoch_loss = totalLoss / epoch_samples
            if phase == 'train':
                trainLoss.append(epoch_loss)
                #writer.add_scalar('loss/train', epoch_loss, epoch)
            else:
                testLoss.append(epoch_loss)
                #writer.add_scalar('loss/test', epoch_loss, epoch)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    output = open('unet_loss.pkl', 'wb')
    pickle.dump(trainLoss, output)
    pickle.dump(testLoss, output)
    output.close()
    plt.figure()
    plt.plot(trainLoss)
    plt.plot(testLoss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('unet_loss' + str(num_epochs)  + '.png')
    return model


optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)
model = train_model(model, optimizer_ft, num_epochs=train_epoch)
#writer.export_scalars_to_json("unet_scalars.json")
#writer.close()
torch.save(model, 'UnetModel.pkl')
torch.save(model.state_dict(), 'UnetModelStateDict.pkl')
plt.show()
