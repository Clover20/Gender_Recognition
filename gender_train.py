import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import default_collate


def loaddata():
    # 路径
    path = "data/data"
    transform = transforms.Compose([
        # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
        transforms.Resize((32, 32)),
        # 在图片的中间区域进行裁
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])

    full_dataset= torchvision.datasets.ImageFolder(path, transform)
    train_size=int(0.8 * len(full_dataset))
    test_size=len(full_dataset)-train_size
    train_dataset,test_dataset = torch.utils.data.random_split(full_dataset,[train_size,test_size])
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              ##每一个batch加载4组样本
                                              batch_size=4,
                                              #每一个epoch之后是否对样本进行随机打乱
                                              shuffle=True,
                                              #几个线程来工作
                                              num_workers=2,
                                              pin_memory=True
                                             )

    testloader = torch.utils.data.DataLoader(test_dataset,
                                              ##每一个batch加载4组样本
                                              batch_size=4,
                                              # 每一个epoch之后是否对样本进行随机打乱
                                              shuffle=True,
                                              # 几个线程来工作
                                              num_workers=2,
                                              pin_memory=True
                                              )
    return trainloader,testloader

# 4层神经网络
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # 输入深度为3，输出为6，卷积核大小为 5*5 的 conv1 变量
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 池化层 2*2窗口  每次滑动2个元素
        self.pool = nn.MaxPool2d(2, 2)
        # 输入深度为6，输出为16，卷积核大小为 5*5 的 conv2 变量
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # 2个输出 男女
        self.fc3 = nn.Linear(84, 2)

    # 前向传播
    def forward(self, x):
        #2次池化 提取特征
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # view()函数用来改变tensor的形状，
        # 例如将2行3列的tensor变为1行6列，其中-1表示会自适应的调整剩余的维度
        # 在CNN中卷积或者池化之后需要连接全连接层，所以需要把多维度的tensor展平成一维
        x = x.view(x.size(0), -1)
        # 从卷基层到全连接层的维度转换
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def trainandsave(dataloader, model, criterion, optimizer,device):

    size = len(dataloader.dataset)
    print(size)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = criterion(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    print(size)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trainloader,testloader = loaddata()
    # 神经网络结构
    net = NeuralNetwork().to(device)
    # 优化器
    # 随机梯度下降 动量为0.9 学习率为0.001
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    epochs = 60
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        trainandsave(trainloader, net, criterion, optimizer, device)
        test(testloader, net, criterion)
    print('Finished Training')
    torch.jit.save(torch.jit.script(net), 'final.pt')



