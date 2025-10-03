import torch
import pickle

def debug_model_loading(weight_path):
    try:
        # Thử load với map_location
        model = torch.load(weight_path, map_location='cpu')
        print("Model loaded successfully!")
        print(f"Model keys: {model.keys()}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # Thử với pickle protocol khác
        try:
            with open(weight_path, 'rb') as f:
                model = pickle.load(f)
            print("Loaded with pickle directly")
            return model
        except Exception as e2:
            print(f"Pickle error: {e2}")
            return None

# Test
debug_model_loading("yolov7-tiny.pt")