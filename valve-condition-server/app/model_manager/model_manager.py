import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_models.data_instance import DataInstance
from model_manager.valve_dataset import ValveDataset
from .valve_cnn import ValveCNN
import torch.nn as nn
from config.settings import settings

class ModelManager:
    def __init__(self):
        self.get_model()
        with open(settings.STATISTICS_FILE, "r") as f:
            statistics = json.load(f)
        self.X_mean = statistics["mean"]
        self.X_std = statistics["std"]
        self.valve_conditions_classes = {
            0: 73,
            1: 80,
            2: 90,
            3: 100
        }

    def get_model(self):
        """if the model is not already saved, the program trains a new one. Else it loads the saved one"""
        pth_files = list(settings.MODEL_DIRECTORY.glob("*.pth"))
        if pth_files:
            model_path = pth_files[0]
            print(f"Loading model: {model_path.name}")
            self.model = ValveCNN(n_classes=4)
            self.model.load_state_dict(torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
            self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.train_model()

    def predict_valve_condition(self, data_instance: DataInstance):
        """predicts the valve condition using the flow and the pressure"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pressure = data_instance.pressure
        flow = data_instance.flow
        # as the flow data is shorter than the pressure data, we interpolate to make them having the same length
        flow = self.interpolate(len(flow), len(pressure), flow)
        x = np.stack([pressure, flow], axis=0)
        x = (x - self.X_mean) / (self.X_std)
        x = torch.tensor(x, dtype=torch.float32).to(device) 
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            predicted_class = torch.argmax(output, dim=1).item()
        predicted_class = self.valve_conditions_classes[predicted_class]
        return predicted_class
    
    def interpolate(self, number_points_flow, number_points_pressure, flow_individual):
        """give the same lenght to both vectors by interpolating"""
        x_old = np.linspace(0, 1, number_points_flow)
        x_new = np.linspace(0, 1, number_points_pressure)
        return np.interp(x_new, x_old, flow_individual)

    def train_model(self):
        """train the model"""
        # Load the data
        pressure = pd.read_table(str(settings.DATA_DIRECTORY / 'PS2.txt'), header=None).values    # shape: (N_cycles, 6000)
        flow = pd.read_table(str(settings.DATA_DIRECTORY / 'FS1.txt'), header=None).values        # shape: (N_cycles, 600)
        profile = pd.read_table(str(settings.DATA_DIRECTORY / 'profile.txt'), header=None)
        valve_condition = profile.iloc[:, 1].values

        # interpolate to make sure that both vectors have the same length
        number_rows, number_points_pressure = pressure.shape
        _, number_points_flow = flow.shape
        flow_interpolated = np.zeros_like(pressure)
        for i in range(number_rows):
            flow_interpolated[i] = self.interpolate(number_points_flow, number_points_pressure, flow[i])
        
        
        X = np.stack([pressure, flow_interpolated], axis=1)
        y = valve_condition
        
        # normalization
        self.X_mean = X.mean(axis=(0, 2), keepdims=True)
        self.X_std = X.std(axis=(0, 2), keepdims=True)
        # we save the mean and std in a file so we don't have to estimate them again
        with open(str(settings.MODEL_DIRECTORY / "statistics.json"), "w") as f:
            json.dump({
                "mean": self.X_mean.tolist(),
                "std": self.X_std.tolist()
            }, f, indent=4)
        X = (X - self.X_mean) / self.X_std

        # split between train and test
        n_train = 2000
        X_train = X[:n_train]
        X_test = X[n_train:]
        y_train = y[:n_train]
        y_test = y[n_train:]
        train_dataset = ValveDataset(X_train, y_train)
        test_dataset = ValveDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)  

        # training loop
        print("training beginning")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ValveCNN(n_classes=4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        num_epoch = settings.NUMBER_EPOCHS
        train_losses = []
        test_losses = []
        for epoch in range(num_epoch):
            model.train()
            running_loss = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                outputs = model(Xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()* Xb.size(0)
            train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for Xb, yb in test_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    outputs = model(Xb)
                    loss = criterion(outputs, yb)
                    running_loss += loss.item() * Xb.size(0)
            test_loss = running_loss / len(test_loader.dataset)
            test_losses.append(test_loss)
            print(f"Epoch {epoch +1}/{num_epoch} - Train loss: {train_loss} - Test loss: {test_loss}")
        torch.save(model.state_dict(), str(settings.MODEL_DIRECTORY / "valve_model.pth"))
        print("training completed")


# Singleton
model_manager_instance = ModelManager()