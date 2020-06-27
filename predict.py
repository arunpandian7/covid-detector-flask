import torch
import torch.nn as nn
from torchvision import models, transforms

import numpy as np
import cv2
from torch.autograd import Variable

PATH = './model/dense_model.pt'
device = torch.device('cpu')
class_names = ['covid19', 'normal']

#Image Transform
test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

#Defining Model Architecture
def CNN_Model(pretrained):
    model = models.densenet121(pretrained=pretrained)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(class_names))
    model.to(torch.device('cpu'))
    return model
inf_model = CNN_Model(pretrained=False)

#Loading the Model Trained Weights
inf_model.to(torch.device('cpu'))
inf_model.load_state_dict(torch.load(PATH, map_location='cpu'))
inf_model.eval()
print('Inference Model Loaded on CPU')

#reading an image
def predict(image):
    if  image.shape[-1] == 4:
        image_cv = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if  image.shape[-1] == 3:
        image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2BGR)
    if  len(image.shape)== 2:
        image_cv = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_tensor = test_transforms(image_cv)
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    print(input.shape)
    input = input.to(torch.device('cpu'))
    out = inf_model(input)
    _, preds = torch.max(out, 1)
    idx = preds.cpu().numpy()[0]
    pred_class = class_names[idx]
    print("Predicted: {}".format(pred_class))
    return pred_class

