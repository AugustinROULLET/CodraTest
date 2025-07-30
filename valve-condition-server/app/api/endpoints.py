from fastapi import APIRouter
from data_models.data_instance import DataInstance
from model_manager.model_manager import model_manager_instance

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

    
    def _train_model(self):
        """launch a model training"""
        self.model_manager.train_model()

    def _predict_valve_condition(self, data: DataInstance):
        """predict the condition of a valve based on the flow and pressure"""
        return self.model_manager.predict_valve_condition(data)
