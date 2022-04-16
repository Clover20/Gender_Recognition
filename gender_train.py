import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

def loadtraindata():
    # 路径
    path = "data/train"
    train = torchvision.datasets.ImageFolder(path,transform=transforms.Compose([
                                                    # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.Resize((32, 32)),
                                                    #在图片的中间区域进行裁
                                                    transforms.CenterCrop(32),
                                                    transforms.ToTensor()])
                                                )

    trainloader = torch.utils.data.DataLoader(train,
                                              ##每一个batch加载4组样本
                                              batch_size=4,
                                              #每一个epoch之后是否对样本进行随机打乱
                                              shuffle=True,
                                              #几个线程来工作
                                              num_workers=2)
    return trainloader

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
    model.train()
    for epoch in range(50):
        # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        # 定义一个变量方便我们对loss进行输出
        print(f"Epoch {epoch + 1}\n-------------------------------")
        running_loss = 0.0
        for i, (X, y) in enumerate(dataloader):
            # enumerate是python的内置函数，既获得索引也获得数据
            # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
            inputs, labels = X.to(device), y.to(device)
            #inputs, labels = X, y
            # 转换数据格式用Variable
            inputs, labels = Variable(inputs), Variable(labels)
            # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
            optimizer.zero_grad()
            # forward + backward + optimize，把数据输进CNN网络
            outputs = model(inputs)
            # 计算损失值
            loss = criterion(outputs, labels)
            # loss反向传播
            loss.backward()
            # 反向传播后参数更新
            optimizer.step()
            # loss累加
            running_loss += loss.item()
            if i % 200 == 199:
                # 然后再除以200，就得到这两百次的平均损失值
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                # 这一个200次结束后，就把runni

    print('Finished Training')
    # 保存整个神经网络的结构和模型参数
    torch.jit.save(torch.jit.script(model), 'final3.pt')




if __name__ == '__main__':

    trainloader = loadtraindata()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 神经网络结构
    net = NeuralNetwork().to(device)
    # 优化器
    # 随机梯度下降 动量为0.9 学习率为0.001
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    trainandsave(trainloader, net, criterion, optimizer, device)
