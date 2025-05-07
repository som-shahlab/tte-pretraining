import unittest
import torch
import numpy as np
from networks import DenseNet121
from utils import remove_prefix_from_state_dict, load_different_model
import os
import sklearn


class TestimageModel(unittest.TestCase):

    def setUp(self):
        """Setup common parameters and create necessary mock objects"""
        self.root_dir = "./data_temp"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(self.root_dir, "ckpt.pth")

        # Simulate labels (binary classification problem)
        self.labels = torch.nn.functional.one_hot(torch.as_tensor([0, 1, 0, 1])).float()

        # Initialize the model (ensure it can be loaded)
        self.model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(
            self.device
        )

    def test_model_loading(self):
        """Test if the model loads correctly with the given pretrained weights."""
        state_dict = torch.load(self.model_path, map_location=self.device)

        # Load model with different state_dict
        loaded_model = load_different_model(self.model, state_dict)

        # Check if the weights are correctly loaded and filtered
        weights = loaded_model.state_dict()

        # Simulate prefix removal and filtering
        if "module." in list(state_dict.keys())[0]:
            pretrained_dict = remove_prefix_from_state_dict(state_dict)
            filtered_dict = {k: v for k, v in pretrained_dict.items() if k in weights}
            pretrain_unique_dict = {
                k: v for k, v in pretrained_dict.items() if k not in weights
            }
            model_unique_dict = {
                k: v for k, v in weights.items() if k not in pretrained_dict
            }
        else:
            filtered_dict = {k: v for k, v in state_dict.items() if k in weights}
            pretrain_unique_dict = {}
            model_unique_dict = {}

        # Test that the unique weights from the pretrained model are correct
        expected_pretrain_unique_weights = [
            "final_layer.weight",
            "final_layer.bias",
            "ln.weight",
            "ln.bias",
            "task_layer.weight",
            "task_layer.bias",
        ]
        for weight in pretrain_unique_dict:
            self.assertIn(
                weight,
                expected_pretrain_unique_weights,
                f"model doesn't load {weight} from TTE-pretrained model",
            )

        # Ensure all model weights are loaded and initialized correctly
        self.assertEqual(
            model_unique_dict,
            {},
            "model should have all weights from the pretrained model, and not initialize any new ones",
        )

    def test_feature_extraction(self):
        """Test if feature extraction during inference generates correct dimensions."""
        # Create dummy input
        input_image = torch.randn(1, 1, 224, 224, 200).to(self.device)

        # Get model output and features
        with torch.no_grad():
            outputs, features = self.model(input_image, return_features=True)

        # Check that the features have the expected dimensions
        self.assertEqual(
            features.shape[1], 1024, "Extracted features should have 1024 dimensions"
        )

    def test_label_features_matching(self):
        """Test if features and labels have matching dimensions."""
        # Simulate train features and labels
        features_train = np.random.randn(10, 1024)  # 10 samples, 1024 features
        labels_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # 10 labels

        # Simulate validation features and labels
        features_val = np.random.randn(5, 1024)  # 5 samples, 1024 features
        labels_val = np.array([0, 1, 0, 1, 0])  # 5 labels

        # Test shape consistency
        self.assertEqual(
            features_train.shape[0],
            labels_train.shape[0],
            "Train features and labels count should match",
        )
        self.assertEqual(
            features_val.shape[0],
            labels_val.shape[0],
            "Validation features and labels count should match",
        )
        self.assertEqual(
            features_train.shape[1], 1024, "Train features should have 1024 dimensions"
        )
        self.assertEqual(
            features_val.shape[1],
            1024,
            "Validation features should have 1024 dimensions",
        )

    def test_binary_labels(self):
        """Test that the labels are binary and match the expected format."""
        labels_train = np.array([0, 1, 0, 1, 0, 1])
        labels_val = np.array([0, 1, 0, 1, 0])

        # Ensure binary labels
        self.assertEqual(
            len(np.unique(labels_train)), 2, "Train labels should be binary"
        )
        self.assertEqual(
            len(np.unique(labels_val)), 2, "Validation labels should be binary"
        )

    def test_linear_model_inference(self):
        """Test if LogisticRegressionCV works with extracted features."""
        features_train = np.random.randn(10, 1024)  # Simulated training features
        labels_train = np.array(
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        )  # Simulated training labels

        # Initialize and fit the linear model
        linear_model = sklearn.linear_model.LogisticRegressionCV(
            penalty="l2", solver="liblinear", cv=2, n_jobs=-1
        )
        linear_model.fit(features_train, labels_train)

        # Simulate validation
        features_val = np.random.randn(5, 1024)
        y_val_proba = linear_model.predict_proba(features_val)[::, 1]

        # Check that the predicted probabilities have correct shape
        self.assertEqual(
            y_val_proba.shape[0],
            features_val.shape[0],
            "Predicted probabilities should match the number of validation samples",
        )


if __name__ == "__main__":
    unittest.main()
