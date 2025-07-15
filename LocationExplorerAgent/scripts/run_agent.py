import os
from locationexploreragent.agent import EnhancedLocationExplorerAgent 
# from agent import EnhancedLocationExplorerAgent

# docker run -v C:\Users\osama\OneDrive\Desktop\Usama_dev\LocationExplorerAgent\pretrained_models:/app/pretrained_models locationexplorerimage

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "pretrained_models")
model_path = os.path.join(MODEL_DIR, "classification_model.joblib")
regression_model_path = os.path.join(MODEL_DIR, "regression_model.joblib")
def main():
    # Initialize the agent with your data path
    agent = EnhancedLocationExplorerAgent(
        "data/2025-4-11-iolp-buildings.xlsx",
        # model_path=r"C:\Users\osama\OneDrive\Desktop\Usama_dev\LocationExplorerAgent\pretrained_models/classification_model.joblib",
        # regression_model_path=r"C:\Users\osama\OneDrive\Desktop\Usama_dev\LocationExplorerAgent\pretrained_models/regression_model.joblib"
        model_path = model_path,
        regression_model_path = regression_model_path
    )
    
    # Run analysis for a location
    results = agent.full_analysis_pipeline(40.7493378, -73.97407369999999, radius_km=50)
    
    # Save models
    # agent.save_model("models/classification_model.joblib")
    # agent.save_regression_model("models/regression_model.joblib")
    # print("Models saved successfully!")

if __name__ == "__main__":
    main()