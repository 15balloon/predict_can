from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import speech

root = Tk()
root.title("Predict Drink")
root.geometry("380x540")
root.resizable(False, False)
img = None

def file_cmd():
    global img
    root.filename = filedialog.askopenfilename(initialdir="/Desktop", title="Open File", filetypes=(("jpg files", "*.jpg"), ("png files", "*.jpg")))
    img = ImageTk.PhotoImage(Image.open(root.filename).resize((200, 300)))
    img_label.configure(image=img)

def pred_cmd():
    if img:
        drink_name = speech.speech(root.filename)
        res_label.config(text=drink_name)
    else:
        res_label.config(text="이미지를 불러와주세요.")
        
file_btn = Button(root, padx=5, pady=5, text="Open File", command=file_cmd)
file_btn.pack()

img_label = Label(root, image=None)
img_label.pack()

res_label = Label(root, text="")
res_label.pack()

pred_btn = Button(root, padx=5, pady=5, text="Predict", command=pred_cmd)
pred_btn.pack()

root.mainloop()