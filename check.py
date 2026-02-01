import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(r"C:\calAI\backend\food_classifier_best.pth", map_location=DEVICE)
print(checkpoint.keys())
