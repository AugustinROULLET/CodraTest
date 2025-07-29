from fastapi import FastAPI, Path
import torch


class EndpointsManager:
    def __init__(self, app: FastAPI):
        self.app =  app
        self.init_routes()
        self.get_model()

    def get_model(self):
        model_dir = Path("../../model")
        pth_files = list(model_dir.glob("*.pth"))
        if pth_files:
            model_path = pth_files[0]  # Prend le premier fichier trouvé
            print(f"Chargement du modèle : {model_path.name}")
            self.model = SimpleValveCNN(n_classes=4)
            self.model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
            self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.train_model()
            
    def init_routes(self):
        @self.app.post("/train")
        def train_model():
            return self.train_model()
        @self.app.post("/predict")
        def predict_valve(data: EntryData):
            return self.predict_valve(data)
    
    def train_model(self):
        print("train the model")

    def predict_valve(self, data: EntryData):
        input_tensor = torch.tensor([features], dtype=torch.float32)
        input_tensor = input_tensor.to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        return {"prediction": predicted_class}