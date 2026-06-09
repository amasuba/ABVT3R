import numpy as np
import os
import matplotlib.pyplot as plt

class BiomassANN:
    """
    Artificial Neural Network (ANN) for my biomass estimation/prediction from first principles\
    
    This class implements the feedforward neural network, building all its components 
    (forward-pass, backward-pass, optimization) 
    """
    
    def __init__(self):
        """
        Initialization of the ANN with empty weights and configurations
        """
        self.weights = []
        self.biases = []
        self.feature_names = None
        self.scaler_mean = None
        self.scaler_std = None
        self.training_hist = {'loss': [], 'val_loss': []}
        
    # ====================================================================
    # Step 1: Feature Extraction
    # ====================================================================
    
    def extract_features_from_reconstruction(self, reconstruction_dir, plant_id):
        """
        Extract geometric and quality features from reconstruction files
        - Read the reconstruction stats text file
        - Parse numerical values from text 
        - Load mesh geometry from .npy files
        - Calculate derived geometric features
        """
        features = {}
        
        # Load the reconstruction stats from text file
        stats_file = f"{reconstruction_dir}/reconstruction_stats_plant_{plant_id}.txt"
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if 'Merged points:' in line:
                        features['merged_points'] = float(line.split(':')[1].replace(',', '').strip())
                    elif 'Vertices:' in line and i > 0 and 'Final' not in lines[i - 1]:
                        features['vertices'] = float(line.split(':')[1].replace(',', '').strip())
                    elif 'Triangles:' in line and i > 0 and 'Final' not in lines[i - 1]:
                        features['triangles'] = float(line.split(':')[1].replace(',', '').strip())
                    elif 'Surface area:' in line:
                        features['surface_area'] = float(line.split(':')[1].strip().split()[0])
                    elif 'Volume' in line:
                        features['volume'] = float(line.split(':')[1].strip().split()[0])
                    elif 'Overall quality:' in line:
                        features['overall_quality'] = float(line.split(':')[1].strip())
                    elif 'Geometric fidelity:' in line:
                        features['geometric_fidelity'] = float(line.split(':')[1].strip())
                    elif 'Surface smoothness:' in line:
                        features['smoothness'] = float(line.split(':')[1].strip())
                        
        # Load vertex data from geometric calculations
        vertices_file = f"{reconstruction_dir}/final_vertices_plant_{plant_id}.npy"
        
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
            if features['bbox_volume'] > 0:
                features['compactness'] = features['volume'] / features['bbox_volume']
            else:
                features['compactness'] = 0
                
            # Centroid height
            features['centroid_height'] = vertices[:, 1].mean()
            
        return features
        
    def prepare_dataset(self, reconstruction_dir, weights_file, selected_features = None):
        """
        Prepare the complete dataset for training
        - Load ground truth weights from text file
        - Extract the features for each plant
        - Create the feature matrix X and target vector y
        - Store the feature names for consistency
        
        Allows to select the features I wanna use or return all if set to None
        """
        # Load weights
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights = {}
        weights_file = f"{script_dir}/{weights_file}"
        with open(weights_file, 'r') as f:
            for line in f:
                if ':' in line:
                    plant, weight = line.strip().split(':')
                    plant_id = int(plant.split('_')[1])
                    weights[plant_id] = float(weight)
                    
        # Extract features for all plants
        all_features_dicts = []
        all_weights = []
        
        for plant_id in sorted(weights.keys()):
            features = self.extract_features_from_reconstruction(reconstruction_dir, plant_id)
            if features:
                all_features_dicts.append(features)
                all_weights.append(weights[plant_id])
            
        if not all_features_dicts:
            raise ValueError("No features extracted. Check reconstruction_dir and plant_ids.")    
        
        # Determine all possible feature names
        if selected_features is None:
            # Collect all unique feature keys across all feature dictionaries
            all_keys = set()
            for f_dict in all_features_dicts:
                all_keys.update(f_dict.keys())
            self.feature_names = sorted(list(all_keys)) # Ensure consistent order
            print(f"No specific features selected. Using all {len(self.feature_names)} available features.")
        else:
            self.feature_names = selected_features
            print(f"Selected {len(selected_features)} features: {selected_features}")

        # Create the feature matrix X using NumPy
        # Initialize X with zeros, then fill it
        X = np.zeros((len(all_features_dicts), len(self.feature_names)))
        
        for i, feature_dict in enumerate(all_features_dicts):
            for j, feature_name in enumerate(self.feature_names):
                X[i, j] = feature_dict.get(feature_name, 0.0) # Use .get() to handle potential missing features, default to 0.0
        
        y = np.array(all_weights)
        
        print(f"Dataset prepared: {len(X)} samples, {len(self.feature_names)} features")
        print(f"Features: {self.feature_names}")
        print(f"Weight range: {y.min():.2f}g - {y.max():.2f}g")
        
        return X, y, self.feature_names
        
    # ==============================================================================
    # Step 2: Activation Functions ( Building blocks for NN)
    # ==============================================================================
    
    def relu(self, x):
        """
        Rectified Linear Unit activation function
        - Introduces non-linearity
        - Computationally efficient
        - Helps avoid vanishing gradient problem
        - Commonly used in hidden layers
        
        MATHEMATICAL DEF:
        ReLU(x) = max(0, x)
        """
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """
        Derivative of ReLU for backpropagation
        - Used in backpropagation to calculate gradients
        
        MATHEMATICAL DEFINITION:
        d(ReLU)/dx = 1 if x > 0, else 0
        """
        return (x > 0).astype(float)
        
    def linear(self, x):
        """
        Linear activation
        - Used in output layer for regression problems
        - Allows for network to output any real number
        
        MATHEMATICAL DEFINITION:
        f(x) = x
        """
        return x
        
    def linear_derivative(self, x):
        """
        Derivative of linear activation
        
        MATHEMATICAL DEFINITION: 
        d(f)/dx = 1
        """
        return np.ones_like(x)
        
    # ==================================================================================
    # Step 3: Weight Initialization
    # ==================================================================================
    
    def he_initialization(self, n_in, n_out):
        """
        He initialization for weight matrices
        - Scales initial weights based on layer size
        - Larger input layers need smaller initial weights
        - Maintains variance of activations across layers
        
        Use if for the following reasons:
        - Designed specifically for ReLU activations
        - Prevents vanishing/exploding gradiesnts
        - Helps network converge faster during training
        
        MATHEMATICAL DEFINITION:
        W approx N(0, sqrt(2/n_in))
        where n_in is the number of input neurons
        """
        return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        
    def initialize_network(self, layer_sizes):
        """
        Initialize all weights and biases for the neural network
        
        NETWORK ARCHITECTURE:
        layer_sizes defines the structure, e.g. [18, 64, 32, 16, 1]
        where:
        - Input layer: 18 features
        - Hidden layer 1: 64 neurons
        - Hidden layer 2: 32 neurons
        - Hidden layer 3: 16 neurons
        - Output layer: 1 neuron (biomass prediction)
        
        Functionallity:
        - For each pair of consecutive layers:
            - Create weight matrix connecting them
            - Create bias vector for the next layer
        - Use He Initialization for weights
        - Initialize biases to zero
        """
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Initialize weights using He initialization
            W = self.he_initialization(layer_sizes[i], layer_sizes[i + 1])
            self.weights.append(W)
            
            # Initialize biases to zero
            b = np.zeros((1, layer_sizes[i + 1]))
            self.biases.append(b)
            
        print(f"Network initialized with architectureL {layer_sizes}")
        print(f"Total parameters: {self.count_parameters()}")
        
    def count_parameters(self):
        """
        Count the total number of trainable parameters (weights + biases)
        """
        total = 0
        for W, b in zip(self.weights, self.biases):
            total += W.size + b.size
            
        return total
        
    # =========================================================================
    # Step 4: Forward propagation
    # =========================================================================
    
    def forward_propagation(self, X):
        """
        Forward pass through the neural network
        
        For each layer the following is conducted:
        - Linear transformation: Z = X @ W + b where
            - X is the input from the previous layer
            - W is the weight matrix
            - b is the bias vector
            - @ is the matrix multiplication operand
        - Non-linear activation: A = activation(Z) 
            - ReLU for hidden layers
            - linear for output layer
        """
        cache = {'A': [X]} # Store activations 
        cache['Z'] = [] # Store pre-activation values
        
        A = X # Current activation starts as input
        
        # Forward through hidden layers
        for i in range(len(self.weights) - 1):
            # linear transformation
            Z = A @ self.weights[i] + self.biases[i]
            cache['Z'].append(Z)
            
            # ReLU activation for hidden layers
            A = self.relu(Z)
            cache['A'].append(A)
            
        # Output layer
        Z = A @ self.weights[-1] + self.biases[-1]
        cache['Z'].append(Z)
        
        output = self.linear(Z) # Linear activation for output
        cache['A'].append(output)
        
        return output, cache
        
    # =========================================================================
    # Step 5: Loss Function
    # =========================================================================
    
    def mean_squared_error(self, y_true, y_pred):
        """
        Mean squared error loss function
        - Heavily penalizes large errors because of squaring
        - Differentiable
        - Same units as target variable squared
        - Standard choice for regression problems
        
        MATHEMATICAL DEFINITION:
        MSE = (1/n) * sum(y_true - y_pred)^2
        """
        n = y_true.shape[0]
        return np.sum((y_true - y_pred) ** 2) / n
    
    def mse_derivatives(self, y_true, y_pred):
        """
        Derivative of MSE with respect to predictions
        - Starting point for backpropagation
        - Tells us how to adjust predictions to reduce loss
        - Gradient flows backward from here through the netweork
        
        MATHEMATICAL DEFINITION:
        d(mse)/d(y_pred) = (2/n) * (y_pred - y_true)
        """
        n = y_true.shape[0]
        return (2 / n) * (y_pred - y_true)
        
    # ============================================================================
    # Step 6: Backward Propagation
    # ============================================================================
    
    def backward_propagation(self, y_true, cache):
        """
        Backward pass calculates gradients for all weights and biases
        - Algorithm for calculating how much each weight contributed to the error
        - Uses chain rule from calculus to propagate gradients backward
        - Tells us how to adjust weights to reduce loss
        
        MATHEMATICAL PROCESS (Layer by layer):
        - Calcualte error gradient: dL/dA = derivative of loss
        - Flow through activation: dL/dZ = dL/dA * activation'(Z)
        - Calculate weight gradient: dL/dW = A_prev^T @ dL/dZ
        - Calculate bias gradient: dL/db = sum(dL/dZ)
        - Calculate gradient for previous layer: dL/dA_prev = dL/dZ @ W^T
        
        For each hidden layer:
        - Repeat steps 2-5 using gradient from next layer
        """
        gradients = {'dW': [], 'db': []}
        n_samples = y_true.shape[0]
        
        # Start with loss gradient at output
        y_pred = cache['A'][-1]
        dA = self.mse_derivatives(y_true, y_pred)
        
        # Backward through output layer
        Z_output = cache['A'][-1]
        dZ = dA * self.linear_derivative(Z_output) # Output uses linear activation
        
        A_prev = cache['A'][-2] # Activation from second-to-last layer
        dW = (A_prev.T @ dZ) # Gradient for output weights
        db = np.sum(dZ, axis = 0, keepdims = True) # Gradient for output biases
        
        # Store gradients
        gradients['dW'].insert(0, dW)
        gradients['db'].insert(0, db)
        
        # Propagate gradient to previous layer
        dA = dZ @ self.weights[-1].T
        
        # Backward through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            # Gradient flows through ReLU activation
            Z = cache['Z'][i]
            dZ = dA * self.relu_derivative(Z)
            
            # Get activation from previous layer
            A_prev = cache['A'][i]
            
            # Calculate gradients for this layer
            dW = (A_prev.T @ dZ)
            db = np.sum(dZ, axis = 0, keepdims = True)
            
            # Store gradients
            gradients['dW'].insert(0, dW)
            gradients['db'].insert(0, db)
            
            # Propagate to previous layer
            if i > 0:
                dA = dZ @ self.weights[i].T
                
        return gradients
        
    # ===================================================================
    # Step 7: Optimization - Adam optimizer (Adaptive learning)
    # ===================================================================
    
    def initialize_adam_optimizer(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        """
        Initialize Adam optimizer parameters
        - Adaptive Moment estimation
        - combines best of two other optimizers
        - Adapts learning rate for each parameter individually
        - Most popular optimizer in deep learning
        
        MATHEMATICAL COMPONENTS:
        - m (first moment): Exponential moving average of gradients
        - v (second moment): Expoential moving average of squared gradients
        - t: Time step counter
        """
        self.adam_params = {
            'learning_rate': learning_rate,
            'beta1': beta1,
            'beta2': beta2,
            'epsilon': epsilon,
            't': 0, # Time step
            'm_W': [np.zeros_like(W) for W in self.weights], # First moment for weights
            'v_W': [np.zeros_like(W) for W in self.weights], # Second moment for weights
            'm_b': [np.zeros_like(b) for b in self.biases], # First moment for biases
            'v_b': [np.zeros_like(b) for b in self.biases] # Second moment for biases
        }
        
    def adam_update(self, gradients):
        """
        Update weights using Adam optimizer
        
        For each parameter theta (weight or bias):
        - Update biased first moment:
            m_t = beta1 * m_{t-1} + (1 - beta1) * gradient
        - Update biased second moment:
            v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2
        - Bias correction (important for early training):
            m_t_hat = m_t / (1 - beta1^t)
            v_t_hat = v_t / (1 - beta2^t)
            
        - Parameter update:
            theta_t = theta_{t-1} - alpha * m_t_hat / (sqrt(v_t_hat) + epsilon)
            where alpha is learning rate
        """
        self.adam_params['t'] += 1
        t = self.adam_params['t']
        
        alpha = self.adam_params['learning_rate']
        beta1 = self.adam_params['beta1']
        beta2 = self.adam_params['beta2']
        epsilon = self.adam_params['epsilon']
        
        # Update each layer's parameters
        for i in range(len(self.weights)):
            # --- Update weights ---
            # First moment (momentum)
            self.adam_params['m_W'][i] = (beta1 * self.adam_params['m_W'][i] + (1 - beta1) * gradients['dW'][i])
            
            # Second moment (adaptive learning rate)
            self.adam_params['v_W'][i] = (beta2 * self.adam_params['v_W'][i] + (1 - beta2) * (gradients['dW'][i] ** 2))
            
            # Bias correction
            m_hat_W = self.adam_params['m_W'][i] / (1 - beta1 ** t)
            v_hat_W = self.adam_params['v_W'][i] / (1 - beta2 ** t)
            
            # Update weights
            self.weights[i] -= alpha * m_hat_W / (np.sqrt(v_hat_W) + epsilon)
            
            # --- Update biases ---
            # First moment
            self.adam_params['m_b'][i] = (beta1 * self.adam_params['m_b'][i] + (1 - beta1) * gradients['db'][i])
            
            # Second moment
            self.adam_params['v_b'][i] = (beta2 * self.adam_params['v_b'][i] + (1 - beta2) * (gradients['db'][i] ** 2))
            
            # Bias correction
            m_hat_b = self.adam_params['m_b'][i] / (1 - beta1 ** t)
            v_hat_b = self.adam_params['v_b'][i] / (1 - beta2 ** t)
            
            # Update biases
            self.biases[i] -= alpha * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
            
    # =================================================================
    # Step 8: Feature Normalization (Preprocessing for better training
    # =================================================================
    
    def fit_scaler(self, X):
        """
        Calculate mean and standard deviation for feature normalization
        - Features have different scales
        - Large features dominate learning
        - Normalization puts all features on equal footing
        - Speeds up convergence significantly
        
        Z-SCORE NORMALIZATION:
        X_normalized = (X - mean) / std
        
        - Transforms each feature to have mean = 0, std = 1
        - Preserves relationships between samples
        - Makes optimization landscape more symmetric
        """
        self.scaler_mean = np.mean(X, axis = 0, keepdims = True)
        self.scaler_std = np.std(X, axis = 0, keepdims = True)
        
        # Prevent division by zero for constant features
        self.scaler_std = np.where(self.scaler_std == 0, 1, self.scaler_std)
        
    def transform(self, X):
        """
        Apply normalization using fitted parameters
        """
        if self.scaler_mean is None:
            raise ValueError("Scaler not fitted! Call fit_scaler first.")
        
        return (X - self.scaler_mean) / self.scaler_std
        
    def inverse_transform(self, X_normalized):
        """
        Convert normalized features back to original scale
        """
        return X_normalized * self.scaler_std + self.scaler_mean
        
    # ==================================================================
    # Step 9: Training Loop (Main loop/Pipeline)
    # ==================================================================
    
    def train(self, X_train, y_train, X_val = None, y_val = None, epochs = 200, batch_size = 4, learning_rate = 0.001, early_stopping_patience = 30, verbose = True):
        """
        Train the neural network
        
        Trianing Process (Each Epoch):
        - Shuffle training data (prevents leanring order)
        - Split into mini_batches
        - For each batch:
            Forward propagation
            Calculate loss
            Backward propagation
            Adam optimizer update
        - Evaluate on validation set
        - Check early stopping criterion
        
        Training Curve:
        Epoch 1: loss = 5.2 (random weights)
        Epoch 50: loss = 1.3 (learning patterns)
        Epoch 100: loss = 0.4 (refining)
        Epoch 150: loss = 0.2 (nearly converged)
        """
        if verbose:
            print(f"\n{'='*60}")
            print("STARTING TRAINING")
            print(f"{'='*60}")
            print(f"Training samples: {X_train.shape[0]}")
            print(f"Features: {X_train.shape[1]}")
            print(f"Epochs: {epochs}")
            print(f"Batch size: {batch_size}")
            print(f"Learning rate: {learning_rate}")
            
        # Normalize features
        self.fit_scaler(X_train)
        X_train_norm = self.transform(X_train)
        
        if X_val is not None:
            X_val_norm = self.transform(X_val)
            
        # Initialization of Adam optimizer
        self.initialize_adam_optimizer(learning_rate)
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(X_train_norm.shape[0])
            X_shuffled = X_train_norm[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            n_batches = int(np.ceil(X_train_norm.shape[0] / batch_size))
            epoch_loss = 0
            
            for batch in range(n_batches):
                start_index = batch * batch_size
                end_index = min((batch + 1) * batch_size, X_train_norm.shape[0])
                
                X_batch = X_shuffled[start_index:end_index]
                y_batch = y_shuffled[start_index:end_index]
                
                # Forward pass
                y_pred, cache = self.forward_propagation(X_batch)
                
                # Calculate loss
                batch_loss = self.mean_squared_error(y_batch, y_pred)
                epoch_loss += batch_loss
                
                # Backward pass
                gradients = self.backward_propagation(y_batch, cache)
                
                # Update weights
                self.adam_update(gradients)
                
            # Average loss for epoch
            epoch_loss /= n_batches
            self.training_hist['loss'].append(epoch_loss)
            
            # Validation
            if X_val is not None:
                y_val_pred, _ = self.forward_propagation(X_val_norm)
                val_loss = self.mean_squared_error(y_val, y_val_pred)
                self.training_hist['val_loss'].append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best weights
                    best_weights = [W.copy() for W in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1
                    
                if patience_counter >=  early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        print(f"Best validation loss: {best_val_loss:.6f}")
                    
                    # Restore best weights
                    self.weights = best_weights
                    self.biases = best_biases
                    break
                    
                if verbose and (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.6f}")
                    
        if verbose:
            print(f"\n{'=' * 60}")
            print("Training Complete")
            print(f"{'=' * 60}")
            print(f"Final training loss: {self.training_hist['loss'][-1]:.6f}")
            if X_val is not None:
                print(f"Final validation loss: {self.training_hist['val_loss'][-1]:.6f}")
                
    # ==================================================================
    # Step 10: Prediction 
    # ==================================================================
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Process:
        - Normalize input using training statistics
        - Forward propagation through network
        - Return predictions 
        """
        if self.scaler_mean is None:
            raise ValueError("Model not trained: Call train() first.")
            
        X_norm = self.transform(X)
        predictions, _ = self.forward_propagation(X_norm)
        
        return predictions
        
    # =================================================================
    # Step 11: Model Evaluation
    # =================================================================
    
    def evaluate(self, X, y_true):
        """
        Evaluate model performance with multiple metrics
        """
        y_pred = self.predict(X)
        
        # R^2 score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
        
    # ==============================================================
    # Step 12: K-Fold Cross-Validation
    # ==============================================================
    
    def k_fold_cross_validation(self, X, y, n_splits = 5, layer_sizes = None, epochs = 200, learning_rate = 0.001, verbose = True):
        """
        K-fold cross-validation for robust model evaluation
        - Split data into K equal parts (folds)
        - Train K times, each time using different fold as validation
        - Each sample appears in validatation exactly once
        
        Usage of K-folds:
        - More reliable than single train/test split
        - Uses all data for both training and validation
        - Reduces variance in performance estimates
        - Essential for small datasets
        
        Process for K = 5:
        Fold 1: [Val][Train][Train][Train][Train]
        Fold 2: [Train][Val][Train][Train][Train]
        Fold 3: [Train][Train][Val][Train][Train]
        Fold 4: [Train][Train][Train][Val][Train]
        Fold 5: [Train][Train][Train][Train][Val]
        
        Each model is trained from scratch on different 80% of data,
        validated on remaining 20%
        """
        if layer_sizes is None:
            layer_sizes = [X.shape[1], 64, 32, 16, 1]
            
        n_samples = X.shape[0]
        fold_size = n_samples // n_splits
        
        cv_results = {
            'r2_scores': [],
            'rmse_scores': [],
            'mae_scores': [],
            'predictions': [],
            'actuals': []
        }
        
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"K-Fold Cross-Validation (K = {n_splits})")
            print(f"{'=' * 60}")
            print(f"Total samples: {n_samples}")
            print(f"Samples per fold: ~{fold_size}")
            
        # Shuffle indices once
        indices = np.random.permutation(n_samples)
        
        for fold in range(n_splits):
            if verbose:
                print(f"\n--- Fold {fold + 1}/{n_splits} ---")
                
            # Create validation indices for this fold
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_splits - 1 else n_samples
            val_indices = indices[val_start:val_end]
            
            # Training indices are everything else
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
            
            # Split data
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices].reshape(-1, 1)
            X_val_fold = X[val_indices]
            y_val_fold = y[val_indices].reshape(-1, 1)
            
            # Create and train new model for this fold
            fold_model = BiomassANN()
            fold_model.feature_names = self.feature_names
            fold_model.initialize_network(layer_sizes)
            
            fold_model.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold, epochs = epochs, learning_rate = learning_rate, verbose = False)
            
            # Evaluate on validation fold
            metrics = fold_model.evaluate(X_val_fold, y_val_fold)
            
            cv_results['r2_scores'].append(metrics['r2'])
            cv_results['rmse_scores'].append(metrics['rmse']) 
            cv_results['mae_scores'].append(metrics['mae'])
            
            # Store predictions
            y_pred = fold_model.predict(X_val_fold)
            cv_results['predictions'].extend(y_pred.flatten())
            cv_results['actuals'].extend(y_val_fold.flatten())
            
            if verbose:
                print(f"R²: {metrics['r2']:.4f}")
                print(f"RMSE: {metrics['rmse']:.4f}")
                print(f"MAE: {metrics['mae']:.4f}g")
            
        if verbose:
            print(f"\n{'=' * 60}")
            print("Cross-Validation Summary")
            print(f"{'=' * 60}")
            print(f"R² = {np.mean(cv_results['r2_scores']):.4f} ± {np.std(cv_results['r2_scores']):.4f}")
            print(f"RMSE = {np.mean(cv_results['rmse_scores']):.4f}g ± {np.std(cv_results['rmse_scores']):.4f}g")
            print(f"MAE = {np.mean(cv_results['mae_scores']):.4f}g ± {np.std(cv_results['mae_scores']):.4f}g")
                
        return cv_results
        
    # =========================================================
    # Step 13: Save and Load Model
    # =========================================================

    def save_model(self, filepath):
        """
        Save trained model to device
        - All weight matrices
        - All bias vectors
        - Normalization parameters
        - Feature names
        """
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'feature_names': self.feature_names
        }
        
        np.save(f"{filepath}.npy", model_data, allow_pickle = True)
        print(f"Model saved to {filepath}.npy")
        
    def load_model(self, filepath):
        """
        Load trained model from disk
        """
        model_data = np.load(f"{filepath}.npy", allow_pickle = True).item()
        
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.scaler_mean = model_data['scaler_mean']
        self.scaler_std = model_data['scaler_std']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filepath}.npy")
        print(f"Architecture: {[W.shape[0] for W in self.weights] + [self.weights[-1].shape[1]]}")
        
    # ==========================================================
    # Step 14: Visualization
    # ==========================================================

    def plot_training_hist(self, save_path = None):
        """
        Plot training and validation loss curves
        """
        plt.figure(figsize = (10, 5))
        
        plt.plot(self.training_hist['loss'], label = 'Training Loss', linewidth = 2)
        if self.training_hist['val_loss']:
            plt.plot(self.training_hist['val_loss'], label = 'validation Loss', linewidth = 2)
            
        plt.xlabel('Epoch', fontsize = 12)
        plt.ylabel('MSE Loss', fontsize = 12)
        plt.title('Training History', fontsize = 14)
        plt.legend(fontsize = 11)
        plt.grid(True, alpha = 0.3)
        
        if save_path:
            plt.savefig(save_path, dpi = 300, bbox_inches = 'tight')
        plt.show()
        
    def plot_predictions(self, X, y_true, save_path = None):
        """
        Plot the predicted vs actual values
        """
        y_pred = self.predict(X).flatten()
        y_true = y_true.flatten()
        
        plt.figure(figsize = (8, 8))
        
        plt.scatter(y_true, y_pred, alpha = 0.6, s = 100)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label = 'Perfect prediction', linewidth = 2)
        
        # Calculate metrics for plot
        metrics = self.evaluate(X, y_true.reshape(-1, 1))
        plt.xlabel('Actual Biomass (g)', fontsize = 12)
        plt.ylabel('Predicted Biomass (g)', fontsize = 12)
        plt.title(f'Biomass Predictions\nR² = {metrics["r2"]:.3f}, RMSE = {metrics["rmse"]:.3f}g', 
                 fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
if __name__ == "__main__":
    print("=" * 70)
    print("Biomass Prediction Using ANN From First Principles")
    print("=" * 70)
    
    # Initialize model
    model = BiomassANN()

    # Select only the most important features (physically meaningful)
    selected_features = [
        'volume',           # Most direct relation to biomass
        'surface_area',     # Plant size indicator
        'height',           # Vertical dimension
        'compactness',      # Density measure
        'overall_quality'   # Reconstruction quality
    ] 

    # Auto-detect paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reconstruction_dir = os.path.join(script_dir, "reconstruction_output")
    weights_file = "weights.txt"  # Will look in reconstruction_output folder

    # Check if paths exist
    if not os.path.exists(reconstruction_dir):
        print(f"ERROR: reconstruction_output folder not found at: {reconstruction_dir}")
        print("Please ensure reconstruction_output/ is in the same directory as this script.")
        exit(1)
    
    weights_path = os.path.join(script_dir, weights_file)
    if not os.path.exists(weights_path):
        print(f"ERROR: weights.txt not found at: {weights_path}")
        exit(1)

    # Load and prepare data
    X, y, feature_names = model.prepare_dataset(
        reconstruction_dir=reconstruction_dir,
        weights_file=weights_file,
        selected_features=selected_features
    )

    # Reshape y for neural network
    y = y.reshape(-1, 1)

    # Define network architecture
    architecture = [X.shape[1], 4, 2, 1]

    print(f"\n{'='*70}")
    print("PERFORMING K-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")

    # Cross-validation
    cv_results = model.k_fold_cross_validation(
        X, y,
        n_splits=5,
        layer_sizes=architecture,
        epochs=200,
        learning_rate=0.001,
        verbose=True
    )

    print(f"\n{'='*70}")
    print("TRAINING FINAL MODEL ON ALL DATA")
    print(f"{'='*70}")

    # Train final model
    model.initialize_network(architecture)
    model.train(
        X, y,
        epochs=200,
        learning_rate=0.001,
        verbose=True
    )

    # Evaluate final model
    final_metrics = model.evaluate(X, y)
    print(f"\nFinal Model Performance:")
    print(f"  R²: {final_metrics['r2']:.4f}")
    print(f"  RMSE: {final_metrics['rmse']:.4f}g")
    print(f"  MAE: {final_metrics['mae']:.4f}g")
    print(f"  MAPE: {final_metrics['mape']:.2f}%")

    # Ensure ANN_model directory exists
    model_dir = os.path.join(script_dir, "ANN_model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save model
    model.save_model(os.path.join(model_dir, "biomass_ann_model"))

    # Plot results
    model.plot_predictions(X, y, save_path=os.path.join(model_dir, 'predictions.png'))
    model.plot_training_hist(save_path=os.path.join(model_dir, 'training_history.png'))

    print(f"\n{'='*70}")
    print("COMPLETE! Model ready for new predictions.")
    print(f"{'='*70}")
        
        
                    
