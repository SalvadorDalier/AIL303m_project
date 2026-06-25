import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

def googlenet_model(num_class=2):
    model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    model.aux_logits = False
    ori_feature = model.fc.in_features
    model.fc = nn.Linear(in_features=ori_feature, out_features=num_class)
    return model 

if __name__ == "__main__":
    googlenet = googlenet_model(num_class=2)
    INPUT_SIZE = (32,3,224,224)

    print('GoogleNET is running')

    summary(
        googlenet,
        input_size = INPUT_SIZE,
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=25,
        row_settings=['var_names']
    )