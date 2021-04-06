import torch
from models import *

net = VGG('VGG19')
checkpoint = torch.load('FER2013_VGG19/PrivateTest_model.t7',map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['net'])
# net.cuda()
net.eval()

dummy_input = Variable(torch.randn(1, 3, 44, 44))
input_names = ["input_1"]
output_names = ["output_1"]
torch.onnx.export(net,
                  dummy_input,
                  "vgg19.onnx",
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)
