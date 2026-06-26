import torch
import gc

from model.googlenet22 import googlenet_model
from model.densenet121 import densenet_model
from dataset import dataloader

# implement far more 3 models
def get_model(name, num_class=2):
    if name == "googlenet22": return googlenet_model(num_class)
    if name == "densenet121": return densenet_model(num_class)

    else: raise ValueError(f'không có model: {name}')

# setup run models 
def benchmark():
    models_name = ["googlenet22", "densenet121"]
    
    # run dataloader
    train_loader, valid_loader = dataloader()

    comparison = []
    for m in models_name:
        print(f'đang chạy model {m.upper()}')
    
        model = get_model(m)
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # run example: 
        # best_acc = 95 if m == "googlenet22" else 10
        best_acc = 95 if m == "googlenet22" else 10
        
        comparison.append({
            "Model": m.upper(),
            "Best_acc": best_acc
        })

        # clean vram after runnning model
        del model 
        gc.collect()
        torch.cuda.empty_cache()
        print('Cleaned RAM')

    print('\n'+'-'*55)
    comparison_results = sorted(comparison, key=lambda x: x['Best_acc'], reverse = True)
    for res in comparison_results:
        print(f"| {res['Model']:<15} | {res['Best_acc']:>31.2f} % |")

if __name__ == "__main__":
    benchmark()
