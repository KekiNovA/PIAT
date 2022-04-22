import tkinter as tk 
from tkinter import filedialog
from face_detection import start
from scipy.spatial.distance import cosine


LARGE_FONT = ("Verdana", 12)

class App(tk.Tk):

  video = None # For videos use list or dictionary
  image = None

  def __init__(self, *args, **kwargs):
    tk.Tk.__init__(self, *args, **kwargs)
    container = tk.Frame(self)

    container.pack(side="top", fill="both", expand=True)

    container.grid_rowconfigure(0, weight=1)
    container.grid_columnconfigure(0, weight=1)

    self.frames = {}

    for F in (StartPage, ResultPage):

      frame = F(container, self)

      self.frames[F] = frame 

      frame.grid(row=0, column=0, sticky="nsew")

    self.show_frame(StartPage) 

  def show_frame(self, cont):

    frame = self.frames[cont]
    frame.tkraise()


class StartPage(tk.Frame):
  
  def __init__(self, parent, controller):
    self.controller = controller

    tk.Frame.__init__(self, parent)

    vid_loc = tk.StringVar()
    vid_loc.set("-")

    upload_vid = tk.Label(self, textvariable=vid_loc)
    upload_vid.pack()

    img_loc = tk.StringVar()
    img_loc.set("-")

    upload_img = tk.Label(self, textvariable=img_loc)
    upload_img.pack()

    up_vid_but = tk.Button(self, text="Upload Video", command=lambda: self.UploadVid(vid_loc))
    up_vid_but.pack()

    up_img_but = tk.Button(self, text="Upload Image", command=lambda: self.UploadImg(img_loc))
    up_img_but.pack()
     
    analyze_but = tk.Button(self, text="Analyze", command=lambda: self.analyze(controller.video, controller.image))
    analyze_but.pack()
  
  def analyze(self, video, image):
    if self.controller.video != None and self.controller.image != None:
      print("Analyzing")
      result = start(video, image)
      self.controller.show_frame(ResultPage)
      self.controller.frames[ResultPage].display(result)
    else:
      tk.messagebox.showinfo("Error", "Please select photo or video.")

  def UploadVid(self, vid_loc):
    location = filedialog.askopenfilename(filetypes=[("MP4 files", ".mp4")])
    self.controller.video = location
    vid_loc.set(location)

  def UploadImg(self, img_loc):
    location = filedialog.askopenfilename(filetypes=[("JPEG files", ".jpg .jpeg")])
    self.controller.image = location
    img_loc.set(location)

class ResultPage(tk.Frame):
  thresh = 0.5
  def __init__(self, parent, controller):
    tk.Frame.__init__(self, parent)
    self.controller = controller

  def display(self, frame_dict):
    for i, time_stamp in enumerate(frame_dict):
      score = cosine(frame_dict["given_face"], frame_dict[time_stamp])
      if score <= self.thresh:
        button= tk.Button(self, text=f"timestamp: {time_stamp}", command=lambda: open_frame(key))
        button.pack()
        #label.bind(f"<Button-{i+1}>", lambda time_stamp: self.open_frame(time_stamp))

  def open_frame(time_stamp):
    print(time_stamp)
         
app = App()
app.title("PIAT")
app.mainloop()

