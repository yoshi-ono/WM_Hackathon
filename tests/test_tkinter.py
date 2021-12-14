import tkinter as tk
from PIL import Image, ImageTk
import threading
from multiprocessing import Process, Queue
import os
import time
import cv2

class TkEye(tk.Frame):
    def __init__(self, root=None):
        super().__init__(root)

        self.create_canvas()

    def create_canvas(self):
        self.label = tk.Label(self, text='Label')
        self.label.pack()

        self.canvas = tk.Canvas(
            self, # 親要素をメインウィンドウに設定
            width=500,  # 幅を設定
            height=500, # 高さを設定
            relief=tk.RIDGE  # 枠線を表示
            # 枠線の幅を設定
        )
        
        #self.canvas.place(x=0, y=0)  # メインウィンドウ上に配置
        self.canvas.pack()

        #PILでjpgを使用
        self.img1 = Image.open('resources/Active vision sample.png')
        #self.img1.thumbnail((500, 500), Image.ANTIALIAS)
        self.img1 = ImageTk.PhotoImage(self.img1)  # 表示するイメージを用意

        self.c_item = self.canvas.create_image(  # キャンバス上にイメージを配置
            0,  # x座標
            0,  # y座標
            image=self.img1,  # 配置するイメージオブジェクトを指定
            tag="illust",  # タグで引数を追加する。
            anchor=tk.NW  # 配置の起点となる位置を左上隅に指定
        )

    def set_image(self, img):    
        self.img1 = img
        self.canvas.itemconfig(self.c_item, image=self.img1)
    
    #def start(self):
    #    self.mainloop()

#----------
# FUNCTION
#----------
def get_image(q=None):
    #PILでjpgを使用
    img1 = Image.open('results.png')
    #self.img1.thumbnail((500, 500), Image.ANTIALIAS)
    img1 = ImageTk.PhotoImage(img1)  # 表示するイメージを用意
    if (q != None):
        img2 = cv2.imread('results.png')
        q.put(img2)
    print('[get_image] process id:', os.getpid())
    return img1

def show1():
    tk_eye1 = TkEye()
    tk_eye1.grid(column=0,row=0)
    #print('[1] parent process:', os.getppid())
    print('[show1] process id:', os.getpid())

def show2(root, q=None):
    #img = get_image()
    if (q != None):
        img = q.get()

    tk_eye2 = TkEye(root)
    #tk_eye2.set_image(img)
    tk_eye2.grid(column=0,row=0)
    #print('[2] parent process:', os.getppid())
    print('[show2] process id:', os.getpid())
    return tk_eye2

def change(tk_eye2, q=None):
    time.sleep(3)
 
    if (q != None):
        img = q.get()
    else:
        img = get_image()
        
    tk_eye2.set_image(img)

#------
# MAIN
#------
if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('800x560')
    root.title('test_tk')

    q = Queue()
    proc = Process(target=get_image, args=(q,))
    proc.start()

    print(__name__, 'process id:', os.getpid())

    #thread1 = threading.Thread(target=start1)
    #thread1.start()

    #p = Process(target=start2, args=(root,))
    #p.start()
    eye = show2(root, q)

    #time.sleep(1.0)
    thread1 = threading.Thread(target=change, args=(eye,q,))
    thread1.start()
   

    root.mainloop()
