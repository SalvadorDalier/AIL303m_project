import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import cv2
from torchvision import transforms

from model.googlenet22 import googlenet_model
from model.densenet121 import densenet_model
from model.resnet50 import resnet50_model
from model.vgg16 import vgg16_model
from model.vgg19 import vgg19_model

def get_model(name, num_class=2):
    if name == "googlenet22": return googlenet_model(num_class)
    if name == "densenet121": return densenet_model(num_class)
    if name == "resnet50": return resnet50_model(num_class)
    if name == "vgg16": return vgg16_model(num_class)
    if name == "vgg19": return vgg19_model(num_class)
    raise ValueError(f"Unknown model: {name}")

def predict_image(img_path, model_name="googlenet22"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Khoi tao model
    model = get_model(model_name)
    project_root = Path(__file__).resolve().parent.parent
    
    if model_name == "googlenet22": weight_name = "googlenet.npy"
    elif model_name == "densenet121": weight_name = "densenet.npy"
    else: weight_name = f"{model_name}.npy"
        
    weight_path = project_root / "weights" / weight_name
    if not weight_path.exists():
        print(f"[-] Khong tim thay trong so (Weights) tai {weight_path}")
        print("Vui long chay file train.py hoac tai tu Cloud ve truoc!")
        return
        
    # Nap trong so
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict.get('state_dict', state_dict))
    model = model.to(device)
    model.eval() # BAT BUOC phai set model.eval() khi test de tat Dropout
    
    # 2. Tien xu ly anh
    if not Path(img_path).exists():
        print(f"[-] Khong tim thay anh tai: {img_path}")
        return
        
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Theem chieu Batch vao anh: [3, 224, 224] -> [1, 3, 224, 224]
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    # 3. Chay du doan va IMPLEMENT HANG SOFTMAX
    with torch.no_grad():
        # Buoc A: Mo hinh in ra Logits (Gia tri tho, chua phai xac suat)
        logits = model(input_tensor)
        
        # Buoc B (IMPLEMENT SOFTMAX): Dung Softmax de ep kieu thanh % xac suat
        probabilities = F.softmax(logits, dim=1)[0]
        
    # 4. In ket qua dep mat
    class_names = ["Healthy", "Unhealthy"]
    
    print("="*60)
    print(f" KET QUA DU DOAN (Mo hinh: {model_name.upper()})")
    print("="*60)
    
    print(f"[-] Logits tho (Chua qua Softmax): {logits[0].cpu().numpy()}")
    print("-" * 60)
    
    print("[*] XAC SUAT (Sau khi qua Softmax):")
    for i, cls in enumerate(class_names):
        prob_percent = probabilities[i].item() * 100
        print(f"    -> {cls:10}: {prob_percent:6.2f}%")
        
    pred_idx = probabilities.argmax().item()
    print("="*60)
    print(f"[+] KET LUAN: Buc anh nay la {class_names[pred_idx].upper()}!")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, help="Duong dan buc anh can kiem tra")
    parser.add_argument('--model', type=str, default="googlenet22", help="Ten mo hinh")
    args = parser.parse_args()
    
    predict_image(args.img, args.model)
