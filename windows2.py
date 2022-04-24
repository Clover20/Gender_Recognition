from tkinter import *
from tkinter import  filedialog
import gender_test as gt
from torch.autograd import Variable
import torch
from PIL import Image, ImageTk

class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.place()
        self.createWidget()

    def createWidget(self):
        global photo
        photo = None
        global wenben
        wenben = None
        self.label03 = Label(self, image=photo)
        self.label03.grid(column=0, row=0)
        # self.label03.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.label04 = Label(self, text=wenben)
        self.label04.grid(column=0, row=0)

        self.btn01 = Button(self, text='打开', command=self.getfile, bg='white', anchor='s')
        self.btn01.grid(column=0, row=1)
        # self.btn01.place(relx=0.8, rely=0.5, anchor=CENTER)

    def getfile(self):
        file_path = filedialog.askopenfilename(title='选择文件', filetypes=[(('JPG', '*.jpg')), ('All Files', '*')])
        img = Image.open(file_path)
        width, height = img.size

        img = img.resize((700, int(700 / width * height)))

        net = gt.reload_net()
        image = gt.image_loader(file_path)

        outputs = net(Variable(image))
        classes = ('男', '女')
        _, predicted = torch.max(outputs.data, 1)
        sex = classes[predicted]



        global photo
        global wenben
        wenben = str(sex)
        photo = ImageTk.PhotoImage(img)  # 实际上是把这个变成全局变量
        self.label03.configure(image=photo)
        self.label03.image = photo
        self.label04.configure(text=wenben)





root = Tk()
root.geometry('700x680')
app = Application(master=root)

root.mainloop()