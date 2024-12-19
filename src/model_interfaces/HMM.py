import numpy as np
import pandas as pd
import os
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer

class HMM:
    def __init__(
        self,
        data_dir="data/processed/",
        models_dir="models/hmm/",
        data_source_name="btc_usdt_30m", 
        model_name="btc_usdt_30m_hmm",
        n_states=3,
        n_components=2
    ):
        """
        Initialize the HMM for financial time series.

        Parameters:
        - data_dir: Directory containing processed data files
        - models_dir: Directory to save model files
        - data_source_name: Name of the data source
        - model_name: Name to save the model as
        - n_states: Number of hidden states
        - n_components: Number of Gaussian components per state
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.data_source_name = data_source_name
        self.model_name = model_name
        
        self.features = ['open', 'high', 'low', 'close', 'volume', 
               'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        self.target = "close"
        
        self.n_states = n_states
        self.n_components = n_components
        self.transition_matrix = np.random.dirichlet(np.ones(n_states), size=n_states)
        self.initial_probabilities = np.random.dirichlet(np.ones(n_states))
        self.gmms = [GaussianMixture(n_components=n_components) for _ in range(n_states)]
        self.imputer = SimpleImputer(strategy='mean')

    def construct_data(self):
        """Load and construct training, validation and test datasets"""
        data_file_name_train = self.data_source_name + "_train_scaled.csv"
        data_file_name_validate = self.data_source_name + "_validate_scaled.csv"
        data_file_name_test = self.data_source_name + "_test_scaled.csv"
        
        # Load data files
        data_train_df = pd.read_csv(os.path.join(self.data_dir, data_file_name_train))
        data_validate_df = pd.read_csv(os.path.join(self.data_dir, data_file_name_validate))
        data_test_df = pd.read_csv(os.path.join(self.data_dir, data_file_name_test))
        
        # Extract features
        data_train_scaled = data_train_df[self.features].values
        data_validate_scaled = data_validate_df[self.features].values
        data_test_scaled = data_test_df[self.features].values
        
        return data_train_scaled, data_validate_scaled, data_test_scaled

    def train(self, max_iter=100):
        """
        Train the HMM using the Expectation-Maximization algorithm.
        Args:
            max_iter: Maximum number of iterations for the EM algorithm.
        """
        # Get training data and impute missing values
        X, _, _ = self.construct_data()
        X = self.imputer.fit_transform(X)
        
        n_samples = len(X)
        responsibilities = np.zeros((n_samples, self.n_states))
        log_likelihoods = []

        # Initialize GMMs by fitting them to random subsets of data
        for state in range(self.n_states):
            random_indices = np.random.choice(n_samples, size=n_samples//self.n_states)
            self.gmms[state].fit(X[random_indices])

        for iteration in range(max_iter):
            # E-step: Compute responsibilities
            for t in range(n_samples):
                state_probs = np.zeros(self.n_states)
                for state in range(self.n_states):
                    state_probs[state] = np.log(self.initial_probabilities[state])
                    state_probs[state] += self.gmms[state].score_samples(X[t].reshape(1, -1))[0]
                
                # Normalize probabilities using log-sum-exp trick for numerical stability
                max_prob = np.max(state_probs)
                responsibilities[t, :] = np.exp(state_probs - max_prob)
                responsibilities[t, :] /= np.sum(responsibilities[t, :])

            # M-step: Update model parameters
            for state in range(self.n_states):
                weighted_data = X * responsibilities[:, state].reshape(-1, 1)
                # Remove any remaining NaN values before fitting
                valid_mask = ~np.isnan(weighted_data).any(axis=1)
                if np.any(valid_mask):
                    self.gmms[state].fit(weighted_data[valid_mask])

            # Update transition matrix using log probabilities
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.transition_matrix[i, j] = np.sum(responsibilities[:-1, i] * responsibilities[1:, j])
                self.transition_matrix[i, :] /= np.sum(self.transition_matrix[i, :])

            # Calculate log likelihood
            log_likelihood = np.sum([np.log(np.sum(responsibilities[t, :])) for t in range(n_samples)])
            log_likelihoods.append(log_likelihood)

            # Check convergence
            if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1e-6:
                break

        # Save model parameters
        os.makedirs(self.models_dir, exist_ok=True)
        model_file = os.path.join(self.models_dir, f"{self.model_name}.joblib")
        joblib.dump({
            'transition_matrix': self.transition_matrix,
            'initial_probabilities': self.initial_probabilities,
            'gmms': self.gmms,
            'imputer': self.imputer,
            'log_likelihoods': log_likelihoods
        }, model_file)

        print(f"HMM model trained successfully. Converged after {iteration+1} iterations.")

    def backtest(self, days=30):
        """
        Backtest the model predictions on test data for specified number of days.
        Args:
            days: Number of days to backtest (default 30)
        Returns:
            Dictionary containing backtest metrics
        """
        # Load test data
        _, _, test_data = self.construct_data()
        
        # Calculate number of 30-minute intervals in specified days
        intervals = days * 48  # 48 thirty-minute intervals per day
        test_window = test_data[-intervals:]
        
        # Get predictions
        predictions = self.predict(test_window)
        
        # Calculate actual price changes
        actual_changes = np.diff(test_window[:, self.features.index('close')])
        actual_trends = ['positive' if change > 0 else 'negative' for change in actual_changes]
        
        # Compare predictions with actual trends
        correct_predictions = sum(1 for i in range(len(actual_trends)) 
                                if predictions['prediction'] == actual_trends[i])
        
        accuracy = correct_predictions / len(actual_trends)
        
        return {
            'accuracy': accuracy,
            'test_period_days': days,
            'total_intervals': intervals,
            'correct_predictions': correct_predictions,
            'predictions': predictions
        }

    def predict(self, X):
        """
        Predict market state probabilities for given data.
        Args:
            X: Features matrix of shape (n_samples, n_features)
        Returns:
            Dictionary containing state probabilities and predicted trend
        """
        # Load saved model
        model_file = os.path.join(self.models_dir, f"{self.model_name}.joblib")
        saved_model = joblib.load(model_file)
        
        self.transition_matrix = saved_model['transition_matrix']
        self.initial_probabilities = saved_model['initial_probabilities']
        self.gmms = saved_model['gmms']
        self.imputer = saved_model['imputer']

        # Impute missing values
        X = self.imputer.transform(X)
        
        n_samples = len(X)
        forward_probs = np.zeros((n_samples, self.n_states))

        # Forward algorithm with numerical stability
        for t in range(n_samples):
            for state in range(self.n_states):
                if t == 0:
                    forward_probs[t, state] = np.log(self.initial_probabilities[state])
                else:
                    forward_probs[t, state] = np.logaddexp.reduce(
                        forward_probs[t-1] + np.log(self.transition_matrix[:, state])
                    )
                forward_probs[t, state] += self.gmms[state].score_samples(X[t].reshape(1, -1))[0]

        # Get final state probabilities
        final_probs = np.exp(forward_probs[-1] - np.logaddexp.reduce(forward_probs[-1]))
        most_likely_state = np.argmax(final_probs)
        
        # Determine market trend prediction
        trend = "positive" if most_likely_state > self.n_states // 2 else "negative"
        confidence = final_probs[most_likely_state]

        return {
            "state_probabilities": final_probs,
            "prediction": trend,
            "confidence": confidence
        }
