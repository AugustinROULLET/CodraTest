from fastapi import APIRouter, HTTPException
import pandas as pd
from data_models.data_instance import DataInstance
from model_manager.model_manager import model_manager_instance
from config.settings import settings
from data_models.index_data import IndexData

class EndpointsManager:
    def __init__(self):
        self.router = APIRouter()
        self._setup_routes()
        self.model_manager = model_manager_instance
        
    def _setup_routes(self):
        """associates routes with the appropriates handlers"""
        self.router.add_api_route(
            f"/train",
            self._train_model,
            methods=["POST"],
        )
        self.router.add_api_route(
            f"/predict",
            self._predict_valve_condition,
            methods=["POST"],
        )
        self.router.add_api_route(
            f"/predict_with_index",
            self._predict_valve_condition_with_index,
            methods=["POST"],
        )
    
    def _train_model(self):
        """launch a model training"""
        self.model_manager.train_model()

    def _predict_valve_condition(self, data: DataInstance):
        """predict the condition of a valve based on the flow and pressure"""
        return self.model_manager.predict_valve_condition(data)
    
    def _predict_valve_condition_with_index(self, index_data: IndexData):
        index = index_data.index
        pressure_df = pd.read_table(str(settings.DATA_DIRECTORY / 'PS2.txt'), header=None)
        flow_df = pd.read_table(str(settings.DATA_DIRECTORY / 'FS1.txt'), header=None)
        if index < 0 or index >= len(pressure_df):
            raise HTTPException(status_code=422, detail="Index out of range.")
        pressure_sample = pressure_df.iloc[index].values
        flow_sample = flow_df.iloc[index].values
        data = DataInstance(pressure=pressure_sample, flow=flow_sample)
        return self.model_manager.predict_valve_condition(data)
