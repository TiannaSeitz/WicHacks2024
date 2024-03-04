import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation
import pandas as pd
from PIL import Image
import cv2
import torch # PyTorch package
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks

# def edit_frame(frame, preds):
#     # frame = transforms.Compose([transforms.ToPILImage()])  
#     filepath = ['cap1.png','led1.png','nope1.png']
#     preds_num = int(preds) # just in case because idk
#     graphic = Image.open(filepath[preds_num])
#     graphic = graphic.load()
#     overlay = cv2.addWeighted(frame, 1, graphic, 0.5, 0)
    # return overlay

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, stride = 1) # fix all deez
        # self.conv11 = nn.Conv2d(32, 32, kernel_size = 3, stride = 1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1)
        # self.conv22 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(8198656, 64) # deploy on Lilys comp
        self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(120, 84)
        # self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 3)

    def forward(self, img):
        img = self.conv1(img)
        # img = self.conv11(img)
        img = self.pool1(img)
        img = self.conv2(img)
        # img = self.conv22(img)
        img = self.pool2(img)
        img = torch.flatten(img, 1)
        img = self.fc1(img)
        img = self.relu1(img)
        # img = self.fc2(img)
        # img = self.relu2(img)
        img = self.fc3(img)
        return img

if __name__ == '__main__':
    path = "your path here"
    saved_weights = torch.load(path, map_location=torch.device('cpu'))
    model = cnn()
    model.load_state_dict(saved_weights)
    model.eval()
    classes = ['capacitor', 'led', 'none']
    # define a video capture object 
    vid = cv2.VideoCapture(0)
    vid.set(3, 1920)
    vid.set(4, 1080)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    while(True): 
        # Capture the video frame 
        # by frame 
        ret, frame = vid.read()
        frame_tensor = transform(frame).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(frame_tensor)
            _, preds = torch.max(outputs, 1)
            # Display the resulting frame 
            # result = edit_frame(frame, preds)
            
        cv2.imshow('Watts That?', frame) 
        print(classes[int(preds)])
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 
