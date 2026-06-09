import numpy as np
import os
from ann_class import BiomassANN

"""
Simple Prediction Script - Predict a single plant by ID
"""

if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION - CHANGE THESE VALUES
    # ========================================================================
    PLANT_ID = 1  # <<< CHANGE THIS to predict different plants
    
    print("="*70)
    print("BIOMASS PREDICTION USING TRAINED ANN MODEL")
    print("="*70)
    
    # Auto-detect paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(script_dir, "ANN_model", "biomass_ann_model")
    RECONSTRUCTION_DIR = os.path.join(script_dir, "reconstruction_output")
    WEIGHTS_FILE = os.path.join(script_dir, "weights.txt")
    
    # Features used during training (MUST MATCH TRAINING)
    SELECTED_FEATURES = [
        'volume',
        'surface_area', 
        'height',
        'compactness',
        'overall_quality'
    ]
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH + ".npy"):
        print(f"ERROR: Model not found at {MODEL_PATH}.npy")
        print("Please train the model first by running ann_class.py")
        exit(1)
    
    if not os.path.exists(RECONSTRUCTION_DIR):
        print(f"ERROR: reconstruction_output folder not found at {RECONSTRUCTION_DIR}")
        exit(1)
    
    # Load trained model
    print("\nLoading trained ANN model...")
    model = BiomassANN()
    model.load_model(MODEL_PATH)
    model.feature_names = SELECTED_FEATURES
    print("Model loaded successfully!\n")
    
    # ========================================================================
    # PREDICT SINGLE PLANT
    # ========================================================================
    print("="*70)
    print(f"PREDICTING PLANT {PLANT_ID}")
    print("="*70)
    
    # Extract features
    features_dict = model.extract_features_from_reconstruction(RECONSTRUCTION_DIR, PLANT_ID)
    
    if not features_dict:
        print(f"\nERROR: Could not extract features for plant {PLANT_ID}")
        print(f"Check if files exist:")
        print(f"  - {RECONSTRUCTION_DIR}/final_vertices_plant_{PLANT_ID}.npy")
        print(f"  - {RECONSTRUCTION_DIR}/reconstruction_stats_plant_{PLANT_ID}.txt")
        exit(1)
    
    # Check if all required features are present
    missing_features = [feat for feat in SELECTED_FEATURES if feat not in features_dict]
    if missing_features:
        print(f"\nERROR: Missing features: {missing_features}")
        print(f"Available features: {list(features_dict.keys())}")
        exit(1)
    
    # Convert to array in correct order
    X = np.array([[features_dict[feat] for feat in SELECTED_FEATURES]])
    
    # Make prediction
    prediction = model.predict(X)[0, 0]
    
    # Load actual weight if available
    actual_weight = None
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, 'r') as f:
            for line in f:
                if ':' in line:
                    plant, weight = line.strip().split(':')
                    pid = int(plant.split('_')[1])
                    if pid == PLANT_ID:
                        actual_weight = float(weight)
                        break
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    print(f"\n{'='*70}")
    print("PREDICTION RESULTS")
    print(f"{'='*70}\n")
    
    print(f"Plant ID: {PLANT_ID}")
    print(f"Predicted Weight: {prediction:.2f}g")
    
    if actual_weight is not None:
        error = prediction - actual_weight
        error_percent = (error / actual_weight) * 100
        
        print(f"Actual Weight: {actual_weight:.2f}g")
        print(f"Error: {error:+.2f}g")
        print(f"Error Percentage: {error_percent:+.1f}%")
        
        # Visual indicator
        if abs(error_percent) < 10:
            print("\n✓ EXCELLENT prediction (within 10%)")
        elif abs(error_percent) < 20:
            print("\n✓ GOOD prediction (within 20%)")
        elif abs(error_percent) < 30:
            print("\n⚠ MODERATE prediction (within 30%)")
        else:
            print("\n✗ POOR prediction (>30% error)")
    else:
        print(f"Actual Weight: NOT AVAILABLE")
        print("\nNote: Add plant weight to weights.txt for error calculation")
    
    # Display features used
    print(f"\n{'='*70}")
    print("FEATURES USED FOR PREDICTION")
    print(f"{'='*70}\n")
    
    for feat in SELECTED_FEATURES:
        print(f"  {feat:<20}: {features_dict[feat]:.6f}")
    
    print(f"\n{'='*70}")
    print("PREDICTION COMPLETE")
    print(f"{'='*70}")
