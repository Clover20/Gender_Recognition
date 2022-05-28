from PIL import Image, ImageTk # 导入图像处理函数库
import tkinter as tk
from tkinter import filedialog   #导入文件对话框函数库
import gender_test as gt
from torch.autograd import Variable
import torch



# 建立窗口 设定大小并命名
window = tk.Tk()
window.title('图像显示界面')
window.geometry('600x500')
global img_png           # 定义全局变量 图像的
var = tk.StringVar()    # 这时文字变量储存器
global img_path
# 建立打开图像和显示图像函数


def Open_Img():
    global img_path
    global img_png
    OpenFile = tk.Tk() #建立新窗口
    OpenFile.withdraw()
    file_path = filedialog.askopenfilename()
    img_path = file_path
    Img = Image.open(file_path)
    img_png = ImageTk.PhotoImage(Img)
    var.set('已打开')
def Show_Img():
    global img_png
    global img_path
    print(type(img_path))

    net = gt.reload_net()
    image = gt.image_loader(img_path)

    outputs = net(Variable(image))
    classes = ('男', '女')
    _, predicted = torch.max(outputs.data, 1)

    print(classes[predicted])

    var.set('已显示')   # 设置标签的文字为 'you hit me'
    label_Img = tk.Label(window, image=img_png)

    label_Img.pack()

# 建立文本窗口，显示当前操做状态
Label_Show = tk.Label(window,
    textvariable=var,   # 使用 textvariable 替换 text, 由于这个能够变化
    bg='blue', font=('Arial', 12), width=15, height=2)
Label_Show.pack()
# 建立打开图像按钮
btn_Open = tk.Button(window,
    text='打开图像',      # 显示在按钮上的文字
    width=15, height=2,
    command=Open_Img)     # 点击按钮式执行的命令
btn_Open.pack()    # 按钮位置
# 建立显示图像按钮
btn_Show = tk.Button(window,
    text='显示图像',      # 显示在按钮上的文字
    width=15, height=2,
    command=Show_Img)     # 点击按钮式执行的命令
btn_Show.pack()    # 按钮位置

# 运行总体窗口
window.mainloop()