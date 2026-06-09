import numpy as np
import os

class DecisionTreeRegressor:
    """
    Decision tree for regression
    
    A decision tree recursively splits data based on feature values to 
    minimize prediction error (MSE)
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        """
        Initialize decision tree parameters
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        
    def fit(self, X, y):
        """
        Build the decision tree
        """
        self.n_features = X.shape[1]
        self.tree = self.build_tree(X, y, depth=0)
        
    def build_tree(self, X, y, depth):
        """
        Recursively build tree structure
        """
        n_samples = X.shape[0]
        
        # Stopping criteria - Create leaf node
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return {'leaf': True, 'value': np.mean(y)}
        
        # Find best split
        best_feature, best_threshold = self.find_best_split(X, y)
        
        # If no valid split found, create leaf
        if best_feature is None:
            return {'leaf': True, 'value': np.mean(y)}
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build children
        left_child = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_child,
            'right': right_child
        }
    
    def find_best_split(self, X, y):
        """
        Find the best feature and threshold to split on
        """
        n_samples, n_features = X.shape
        
        if n_samples <= self.min_samples_split:
            return None, None
        
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate MSE for this split
                left_mse = np.var(y[left_mask]) * np.sum(left_mask)
                right_mse = np.var(y[right_mask]) * np.sum(right_mask)
                total_mse = (left_mse + right_mse) / n_samples
                
                # Update best split if this is better
                if total_mse < best_mse:
                    best_mse = total_mse
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
        
    def predict(self, X):
        """
        Make predictions for samples
        """
        return np.array([self.predict_sample(sample, self.tree) for sample in X])
        
    def predict_sample(self, sample, node):
        """
        Predict a single sample by traversing tree
        """
        # If leaf node, return value
        if node['leaf']:
            return node['value']
        
        # Compare feature value to threshold
        if sample[node['feature']] <= node['threshold']:
            return self.predict_sample(sample, node['left'])
        else:
            return self.predict_sample(sample, node['right'])

            
class RandomForestRegressor:
    """
    Random Forest Regressor
    
    An ensemble of decision trees trained on random subsets of data
    with random feature selection at each split.
    """
    
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, max_features='sqrt', random_state=42):
        """
        Initialize Random Forest parameters
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_names = None
        
    def fit(self, X, y):
        """
        Train the Random Forest
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.trees = []
        
        print(f"Training Random Forest with {self.n_trees} trees...")
        
        for i in range(self.n_trees):
            # Bootstrap sampling: sample with replacement
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Build tree on bootstrap sample
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
            if (i + 1) % 20 == 0:
                print(f"  Built {i + 1}/{self.n_trees} trees")
        
        print(f"Random Forest training complete!")
        
    def predict(self, X):
        """
        Make predictions by averaging predictions from all trees
        """
        # Get predictions from each tree
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Average predictions
        return np.mean(tree_predictions, axis=0)
        
    def feature_importance(self):
        """
        Calculate feature importance (simplified version)
        """
        if self.feature_names is None:
            return None
        
        # This is a placeholder - proper implementation would track splits
        return {name: 1.0/len(self.feature_names) for name in self.feature_names}


class BiomassRandomForest:
    """
    Wrapper class for biomass prediction using Random Forest
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def extract_features_from_reconstruction(self, reconstruction_dir, plant_id):
        """
        Extract features from reconstruction files
        """
        features = {}
        
        # Load reconstruction stats from text file
        stats_file = os.path.join(reconstruction_dir, f'reconstruction_stats_plant_{plant_id}.txt')
        
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if 'Merged points:' in line:
                        features['merged_points'] = float(line.split(':')[1].replace(',', '').strip())
                    elif 'Vertices:' in line and i > 0 and 'Final' not in lines[i-1]:
                        features['vertices'] = float(line.split(':')[1].replace(',', '').strip())
                    elif 'Triangles:' in line and i > 0 and 'Final' not in lines[i-1]:
                        features['triangles'] = float(line.split(':')[1].replace(',', '').strip())
                    elif 'Surface area:' in line:
                        # Extract just the number, handling any encoding of m² symbol
                        value_str = line.split(':')[1].strip()
                        # Find the first space and take everything before it as the number
                        features['surface_area'] = float(value_str.split()[0])
                    elif 'Volume:' in line:
                        # Extract just the number, handling any encoding of m³ symbol
                        value_str = line.split(':')[1].strip()
                        features['volume'] = float(value_str.split()[0])
                    elif 'Overall quality:' in line:
                        features['overall_quality'] = float(line.split(':')[1].strip())
                    elif 'Geometric fidelity:' in line:
                        features['geometric_fidelity'] = float(line.split(':')[1].strip())
                    elif 'Surface smoothness:' in line:
                        features['smoothness'] = float(line.split(':')[1].strip())
        
        # Load vertex data for geometric calculations
        vertices_file = os.path.join(reconstruction_dir, f'final_vertices_plant_{plant_id}.npy')
        
        if os.path.exists(vertices_file):
            vertices = np.load(vertices_file)
            
            # Height features
            features['height'] = vertices[:, 1].max() - vertices[:, 1].min()
            features['height_mean'] = vertices[:, 1].mean()
            features['height_std'] = vertices[:, 1].std()
            
            # Width features
            features['width_x'] = vertices[:, 0].max() - vertices[:, 0].min()
            features['width_z'] = vertices[:, 2].max() - vertices[:, 2].min()
            features['max_width'] = max(features['width_x'], features['width_z'])
            
            # Bounding box volume
            features['bbox_volume'] = (features['height'] * features['width_x'] * features['width_z'])
            
            # Compactness: How much of the bounding box is filled
            if features.get('volume', 0) > 0 and features['bbox_volume'] > 0:
                features['compactness'] = features['volume'] / features['bbox_volume']
            else:
                features['compactness'] = 0
            
            # Surface to volume ratio - indicates plant density/structure
            if features.get('volume', 0) > 0 and features.get('surface_area', 0) > 0:
                features['surface_to_volume_ratio'] = features['surface_area'] / features['volume']
            else:
                features['surface_to_volume_ratio'] = 0
            
            # Height to volume ratio - captures plant structure (tall vs bushy)
            if features.get('volume', 0) > 0 and features['height'] > 0:
                features['height_to_volume_ratio'] = features['height'] / features['volume']
            else:
                features['height_to_volume_ratio'] = 0
            
            # Centroid height
            features['centroid_height'] = vertices[:, 1].mean()
        
        return features if features else None
        
    def prepare_training_data(self, reconstruction_dir, weights_file, selected_features=None):
        """
        Prepare training data from reconstruction files and weights
        MODIFIED: No pandas dependency
        """
        # Load weights
        weights = {}
        with open(weights_file, 'r') as f:
            for line in f:
                if ':' in line:
                    plant, weight = line.strip().split(':')
                    plant_id = int(plant.split('_')[1])
                    weights[plant_id] = float(weight)
                    
        # Extract features for all plants
        all_features = []
        all_weights = []
        
        for plant_id in sorted(weights.keys()):
            features = self.extract_features_from_reconstruction(reconstruction_dir, plant_id)
            if features:
                all_features.append(features)
                all_weights.append(weights[plant_id])
        
        # Convert to numpy arrays - NO PANDAS
        if selected_features is not None:
            # Use only selected features
            self.feature_names = selected_features
            X = np.array([[feat_dict[fname] for fname in selected_features] 
                         for feat_dict in all_features])
            print(f"Selected {len(selected_features)} features: {selected_features}")
        else:
            # Use all features
            self.feature_names = list(all_features[0].keys())
            X = np.array([[feat_dict[fname] for fname in self.feature_names] 
                         for feat_dict in all_features])
        
        y = np.array(all_weights)
        
        print(f"Dataset prepared: {len(X)} samples, {len(self.feature_names)} features")
        print(f"Features: {self.feature_names}")
        print(f"Weight range: {y.min():.2f}g - {y.max():.2f}g")
        
        return X, y, self.feature_names
        
    def train(self, X, y, n_trees=100, max_depth=5, min_samples_split=2):
        """
        Train Random Forest model
        """
        self.model = RandomForestRegressor(
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        self.model.fit(X, y)
        
    def train_with_outlier_penalty(self, X, y, n_trees = 50, max_depth = 3, min_samples_split = 2, outlier_penalty = 0.2):
        """
        Train with automatic outlier downweightiung using IQR method
        outlier_penalty: weight for outliers (0.0 = ignore completely, 1.0 = normal weight)
        REcommendatiuon: 0.1 to 0.3
        """
        # Detect outlier using IQR
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Calculate weights
        sample_weights = np.ones(len(y))
        outlier_mask = (y < lower_bound) | (y > upper_bound)
        sample_weights[outlier_mask] = outlier_penalty
        
        # Display my outliers 
        print(f"\nOutlier Detection (IQR method):")
        print(f"  Detected {np.sum(outlier_mask)} outlier(s)")
        for i, (val, is_outlier) in enumerate(zip(y, outlier_mask)):
            if is_outlier:
                print(f"  Sample {i+1}: {val:.2f}g (downweighted to {outlier_penalty})")
                
        # Train with weighted sampling
        np.random.seed(42)
        n_samples = X.shape[0]
        self.model = RandomForestRegressor(n_trees = n_trees, max_depth = max_depth, min_samples_split = min_samples_split)
        self.model.trees = []
        
        # Normalize weights
        probabilities = sample_weights / np.sum(sample_weights)
        
        for i in range(n_trees):
            # Weighted bootsrap sampling
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace = True, p = probabilities)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            tree = DecisionTreeRegressor(max_depth = max_depth, min_samples_split = min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.model.trees.append(tree)
            
            if (i + 1) % 20 == 0:
                print(f"    Built {i + 1}/{n_trees} trees")
                
        print(f"Training complete with outlier penatly!")
        
    def predict(self, X):
        """
        Make predictions
        """
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        return self.model.predict(X)
        
    def evaluate(self, X, y_true):
        """
        Evaluate model performance
        """
        y_pred = self.predict(X)
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape}
        
    def k_fold_cross_validation(self, X, y, n_splits=5, n_trees=100, max_depth=5):
        """
        K-Fold cross-validation
        """
        n_samples = X.shape[0]
        fold_size = n_samples // n_splits
        
        cv_results = {'r2_scores': [], 'rmse_scores': [], 'mae_scores': []}
        
        print(f"{'='*70}")
        print(f"K-FOLD CROSS-VALIDATION (K={n_splits})")
        print(f"{'='*70}\n")
        
        indices = np.random.permutation(n_samples)
        
        for fold in range(n_splits):
            print(f"--- Fold {fold+1}/{n_splits} ---")
            
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_splits - 1 else n_samples
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
            
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_val_fold = X[val_indices]
            y_val_fold = y[val_indices]
            
            fold_model = RandomForestRegressor(n_trees=n_trees, max_depth=max_depth)
            fold_model.fit(X_train_fold, y_train_fold)
            
            y_pred = fold_model.predict(X_val_fold)
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((y_val_fold - y_pred) ** 2))
            mae = np.mean(np.abs(y_val_fold - y_pred))
            ss_res = np.sum((y_val_fold - y_pred) ** 2)
            ss_tot = np.sum((y_val_fold - np.mean(y_val_fold)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            cv_results['r2_scores'].append(r2)
            cv_results['rmse_scores'].append(rmse)
            cv_results['mae_scores'].append(mae)
            
            print(f"  R²: {r2:.4f}, RMSE: {rmse:.2f}g, MAE: {mae:.2f}g\n")
        
        # Print summary
        print(f"{'='*70}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"Mean R²: {np.mean(cv_results['r2_scores']):.4f} (±{np.std(cv_results['r2_scores']):.4f})")
        print(f"Mean RMSE: {np.mean(cv_results['rmse_scores']):.2f}g (±{np.std(cv_results['rmse_scores']):.2f}g)")
        print(f"Mean MAE: {np.mean(cv_results['mae_scores']):.2f}g (±{np.std(cv_results['mae_scores']):.2f}g)")
        
        return cv_results
        
    def leave_one_out_cv(self, X, y, n_trees=50, max_depth=3):
        """
        Leave-One-Out Cross-Validation
        """
        n_samples = X.shape[0]
        predictions = []
        actuals = []
        
        print(f"{'='*70}")
        print(f"LEAVE-ONE-OUT CROSS-VALIDATION (N={n_samples})")
        print(f"{'='*70}\n")
        
        for i in range(n_samples):
            # Leave one out
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i, axis=0)
            X_test = X[i:i+1]
            y_test = y[i:i+1]
            
            # Train model
            model = RandomForestRegressor(
                n_trees=n_trees,
                max_depth=max_depth,
                min_samples_split=2
            )
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)[0]
            predictions.append(y_pred)
            actuals.append(y_test[0])
            
            print(f"Sample {i+1}: Actual={y_test[0]:.2f}g, Predicted={y_pred:.2f}g, Error={abs(y_test[0]-y_pred):.2f}g")
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((actuals - predictions)**2))
        mae = np.mean(np.abs(actuals - predictions))
        ss_res = np.sum((actuals - predictions)**2)
        ss_tot = np.sum((actuals - np.mean(actuals))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"\n{'='*70}")
        print("LEAVE-ONE-OUT CV SUMMARY")
        print(f"{'='*70}")
        print(f"R²: {r2:.4f}")
        print(f"RMSE: {rmse:.2f}g")
        print(f"MAE: {mae:.2f}g")
        
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'predictions': predictions, 'actuals': actuals}
    
    def leave_one_out_cv_with_penalty(self, X, y, n_trees = 50, max_depth = 3, outlier_penalty = 0.2):
        """
        Leave-One-Out Cross-Validation with outlier penalty
        """
        n_samples = X.shape[0]
        predictions = []
        actuals = []
        
        print(f"{'='*70}")
        print(f"LEAVE-ONE-OUT CROSS-VALIDATION (N={n_samples})")
        print(f"{'='*70}\n")
        
        for i in range(n_samples):
            # Leave one out
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i, axis=0)
            X_test = X[i:i+1]
            y_test = y[i:i+1]
            
            # Create temporary model for the fold
            fold_model = BiomassRandomForest()
            fold_model.feature_names = self.feature_names
            fold_model.train_with_outlier_penalty(X_train, y_train, n_trees = n_trees, max_depth = max_depth, outlier_penalty = outlier_penalty)
            
            y_pred = fold_model.predict(X_test)[0]
            predictions.append(y_pred)
            actuals.append(y_test[0])
            
            print(f"Sample {i+1}: Actual={y_test[0]:.2f}kg, Predicted={y_pred:.2f}kg, Error={abs(y_test[0]-y_pred):.2f}kg")
            
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((actuals - predictions)**2))
        mae = np.mean(np.abs(actuals - predictions))
        r2 = 1 - np.sum((actuals - predictions)**2) / np.sum((actuals - np.mean(actuals))**2)
        
        print(f"\n{'='*70}")
        print(f"R²: {r2:.4f} | RMSE: {rmse:.2f}kg | MAE: {mae:.2f}kg")
        
        return {'r2': r2, 'rmse': rmse, 'mae': mae}
        
    def save_model(self, filepath):
        """
        Save trained model (simplified - just save parameters)
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'trees': self.model.trees,
            'n_trees': self.model.n_trees,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'random_state': self.model.random_state,
            'feature_names': self.feature_names
        }
        
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        np.save(f"{filepath}.npy", model_data, allow_pickle=True)
        print(f"Model saved to {filepath}.npy")
        
    def load_model(self, filepath):
        """
        Load trained model (simplified)
        """
        if not os.path.exists(f"{filepath}.npy"):
            raise FileNotFoundError(f"Model file not found: {filepath}.npy")
        
        model_data = np.load(f"{filepath}.npy", allow_pickle=True).item()
        
        # Reconstruct the RandomForestRegressor
        self.model = RandomForestRegressor(
            n_trees=model_data['n_trees'],
            max_depth=model_data['max_depth'],
            min_samples_split=model_data['min_samples_split'],
            random_state=model_data['random_state']
        )
        
        # Restore the trained trees
        self.model.trees = model_data['trees']
        
        # Restore feature names
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filepath}.npy")
        print(f"  Trees: {len(self.model.trees)}")
        print(f"  Max depth: {self.model.max_depth}")
        print(f"  Features: {self.feature_names}")


# ======================================================================
# Usage Example
# ======================================================================

if __name__ == "__main__":
    print("="*70)
    print("BIOMASS PREDICTION USING RANDOM FOREST FROM FIRST PRINCIPLES")
    print("="*70)
    
    # Initialize
    rf_model = BiomassRandomForest()
    
    # Auto-detect paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reconstruction_dir = os.path.join(script_dir, "reconstruction_output")
    weights_file = os.path.join(script_dir, "weights.txt")
    
    # Check if paths exist
    if not os.path.exists(reconstruction_dir):
        print(f"ERROR: reconstruction_output folder not found at: {reconstruction_dir}")
        print("Please ensure reconstruction_output/ is in the same directory as this script.")
        exit(1)
    
    if not os.path.exists(weights_file):
        print(f"ERROR: weights.txt not found at: {weights_file}")
        exit(1)
    
    # Prepare data with Option 1 features
    selected_features = [
        'volume',                      # Voxel volume (realistic, physically accurate)
        'surface_area',                # Surface area of mesh
        'height',                      # Plant height
        'bbox_volume',                 # Bounding box volume (proxy for overall size)
        'surface_to_volume_ratio',     # Indicates plant density/structure
        'height_to_volume_ratio'       # Captures plant structure (tall vs bushy)
    ]
    X, y, feature_names = rf_model.prepare_training_data(
        reconstruction_dir,
        weights_file,
        selected_features=selected_features
    )
    
    # Cross-validation
    #cv_results = rf_model.k_fold_cross_validation(X, y, n_splits=5, n_trees=100, max_depth=5)
    
    # Leave out one
    loo_results = rf_model.leave_one_out_cv(X, y, n_trees = 50, max_depth = 3)
    
    # Leave out one with penalty
    loo_penalty = rf_model.leave_one_out_cv_with_penalty(X, y, n_trees = 50, max_depth = 3, outlier_penalty = 0.2)
    
    # LOO with 10% penalty
    loo_strong = rf_model.leave_one_out_cv_with_penalty(X, y, n_trees = 50, max_depth = 3, outlier_penalty = 0.1)
    
    # LOO with 15%
    loo_strong2 = rf_model.leave_one_out_cv_with_penalty(X, y, n_trees = 50, max_depth = 3, outlier_penalty = 0.15)
    
    # LOO with 5%
    loo_strong3 = rf_model.leave_one_out_cv_with_penalty(X, y, n_trees = 50, max_depth = 3, outlier_penalty = 0.05)
    
    # Comparison
    results = {
        'Baseline': loo_results['r2'],
        'Penalty_20': loo_penalty['r2'],
        'Penalty_10': loo_strong['r2'],
        'Penalty_15': loo_strong2['r2'],
        'Penalty_05': loo_strong3['r2']
    }
    
    # Find best R2 and use that method
    best_method = max(results, key = results.get)
    best_r2 = results[best_method]
    
    # Map name to correct penalty value
    penalty_map = {
        'Baseline': 1.0,
        'Penalty_20': 0.2,
        'Penalty_10': 0.1,
        'Penalty_15': 0.15,
        'Penalty_05': 0.05
    }
    
    best_penalty = penalty_map[best_method]
    
    # Train final model
    print(f"{'='*70}")
    print("TRAINING FINAL MODEL ON ALL DATA")
    print(f"{'='*70}\n")
    
    if best_penalty < 1.0:
        rf_model.train_with_outlier_penalty(X, y, n_trees=50, max_depth=3, outlier_penalty=best_penalty)
    else:
        rf_model.train(X, y, n_trees=50, max_depth=3, min_samples_split = 3)
    
    # Evaluate
    metrics = rf_model.evaluate(X, y)
    print(f"\nFinal Model Performance:")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}g")
    print(f"  MAE: {metrics['mae']:.4f}g")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Penalty: {best_penalty:.2f}")
    
    # Feature importance
    importances = rf_model.model.feature_importance()
    if importances and rf_model.feature_names:
        print(f"\nFeature Importances:")
        importance_dict = dict(zip(rf_model.feature_names, 
                                  [importances.get(i, 0) for i in range(len(rf_model.feature_names))]))
        for feat, imp in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.4f}")
    
    # Ensure RF_model directory exists
    model_dir = os.path.join(script_dir, "RF_model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save
    rf_model.save_model(os.path.join(model_dir, "biomass_rf_model"))
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
