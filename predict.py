import torch
import cv2 as cv
import numpy as np

from model import Net
from torchvision import transforms


model_file = "emnist_cnn.pt"
test_image = "1_img.png"
mapping_file = "emnist_mapping.txt"

def get_mapping():
    mapping = {}
    with open(mapping_file) as f:
        for line in f:
            k, v = line.split(' ')
            mapping[int(k)] = int(v)
    return mapping

def predict(images):
    mapping = get_mapping()

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28,28)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor_list = []
    for img in images:
        img_tensor = transform(img)
        img_tensor_list.append(img_tensor)
    device = torch.device("cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load(model_file), strict=False)
    model.eval()
    
    input_tensor = torch.stack(img_tensor_list)
    input_tensor = input_tensor.permute(0, 1, 3, 2)
    result = model(input_tensor)
    # print(result)
    # print(torch.argmax(result, dim=1))
    result_idx = torch.argmax(result,dim=1).detach().numpy()
    result_numpy = torch.softmax(result, dim=1).detach().numpy()
    for i, idx in enumerate(result_idx):
        confidence = (result_numpy[i][idx])
        if confidence:
            print(chr(mapping[idx]), confidence)

def main():
    img = cv.imread(test_image)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    predict([img])

if __name__ == '__main__':
    main()