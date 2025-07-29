from fastapi import FastAPI
import torch

app = FastAPI()


if __name__ == "__main__":
    model = SimpleValveCNN(n_classes=4)
    model.load_state_dict(torch.load("simple_valve_cnn.pth"))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')