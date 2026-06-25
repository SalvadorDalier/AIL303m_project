import torch 
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary
num_class = 2
def densenet_model(num_class):
    weight = models.DenseNet121_Weights.DEFAULT # mặc định thấy trọng số tốt nhất
    model = models.densenet(weight)

    ori_feature = model.classifier.in_features
    model.classifier = nn.Linear(in_features=ori_feature, out_features=num_class)
    return model

if __name__ == "__main__":
    densenet_for_project = densenet_model(num_class)
    INPUT_SIZE = (32, 3, 224, 224)
    print('densenet is running')
    
    summary(
        densenet_for_project,
        input_size=INPUT_SIZE,
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=25,
        row_settings=['var_names']
    )