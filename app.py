#Library Functions
import io
import cv2
import string
import numpy as np
import torch
import torch.nn as nn

from flask import Flask, jsonify, render_template, request, redirect
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

MODEL_PATH = './model/dense_model.pt'
device = torch.device('cpu')
class_names = ['covid19', 'normal']


#Defining Model Architecture
def CNN_model(pretrained):
    inf_model = models.densenet121(pretrained=pretrained)
    num_ftrs = inf_model.classifier.in_features
    inf_model.classifier = nn.Linear(num_ftrs, len(class_names))
    inf_model.to(torch.device('cpu'))
    return inf_model
inf_model = CNN_model(pretrained=False)

#Loading the Model Trained Weights
inf_model.to(torch.device('cpu'))
inf_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
inf_model.eval()
print('Inference model Loaded on CPU')

#Image Transform
def transform_image(image_bytes):
	test_transforms = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((150,150)),
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
	])
	image = Image.open(io.BytesIO(image_bytes))
	image = np.array(image)
	if  image.shape[-1] == 4:
		image_cv = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
	if  image.shape[-1] == 3:
		image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	if  len(image.shape)== 2:
		image_cv = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	return test_transforms(image_cv).unsqueeze(0)

def get_prediction(image_bytes):
	tensor = transform_image(image_bytes=image_bytes)
	outputs = inf_model.forward(tensor)
	_, prediction = torch.max(outputs, 1)
	return class_names[prediction]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			return redirect(request.url)
		file = request.files.get('file')
		if not file:
			return
		img_bytes = file.read()
		prediction_name = get_prediction(img_bytes)
		return render_template('result.html', name=prediction_name.lower())

	return render_template('index.html')

if __name__ == '__main__':
	# app.run(threaded=True, port=5000)
	app.run(debug=True)
