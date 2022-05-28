import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



classes = ('男','女')
mbatch_size = 8
path = "path"


def image_loader(image_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        # 在图片的中间区域进行裁
        transforms.CenterCrop(32),
        transforms.ToTensor()])
    image = Image.open(image_name)
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)

def loadtestdata():

    testset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((32, 32)),

                                                    transforms.ToTensor()])
                                                )
    testloader = torch.utils.data.DataLoader(testset, batch_size=mbatch_size,
                                             shuffle=True, num_workers=2)
    return testloader

def reload_net():
    trainednet = torch.jit.load('final.pt')
    return trainednet

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test():
    testloader = loadtestdata()
    net = reload_net()
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # nrow是每行显示的图片数量，缺省值为8
    imshow(torchvision.utils.make_grid(images,nrow=4))
    # 打印前25个GT（test集里图片的标签）
    print('真实值: ', " ".join('%5s' % classes[labels[j]] for j in range(mbatch_size)))
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    # 预测值
    print('预测值: ', " ".join('%5s' % classes[predicted[j]] for j in range(mbatch_size)))

def test2():
    testloader = loadtestdata()
    net = reload_net()
    dataiter = iter(testloader)
    images,x = dataiter.next()
    imshow(torchvision.utils.make_grid(images, nrow=5))
    outputs = net(Variable(images).cuda())
    _, predicted = torch.max(outputs.data, 1)
    # 预测值
    print(predicted)
    print('预测值: ', " ".join('%5s' % classes[predicted[j]] for j in range(mbatch_size)))

def test3():
    path = 'path\\images.jpg'

    net = reload_net()
    image = image_loader(path)

    outputs = net(Variable(image))

    _, predicted = torch.max(outputs.data, 1)

    print(classes[predicted])

    outputs = net(Variable(image))

    _, predicted = torch.max(outputs.data, 1)

    print(classes[predicted])



if __name__ == '__main__':
    test3()