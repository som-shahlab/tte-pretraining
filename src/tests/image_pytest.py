import pytest
import torch
import numpy as np
from monai.data import ImageDataset
from monai.transforms import Compose, EnsureChannelFirst, ScaleIntensity, Resize
from networks import DenseNet121
from .utils import load_different_model, remove_prefix_from_state_dict
import os


@pytest.fixture
def load_real_data():
    """Fixture to load real MRI data and labels."""
    root_dir = "./data_temp"

    # Real image file paths (make sure these files exist)
    images = [
        os.path.join(root_dir, "ixi", "IXI314-IOP-0889-T1.nii.gz"),
        os.path.join(root_dir, "ixi", "IXI249-Guys-1072-T1.nii.gz"),
        os.path.join(root_dir, "ixi", "IXI609-HH-2600-T1.nii.gz"),
        os.path.join(root_dir, "ixi", "IXI173-HH-1590-T1.nii.gz"),
        os.path.join(root_dir, "ixi", "IXI020-Guys-0700-T1.nii.gz"),
    ]

    # Corresponding binary labels (0 or 1, e.g., for gender classification)
    labels = np.array([0, 0, 0, 1, 0])  # Modify according to your real data

    # Preprocess data: Apply transformations
    train_transforms = Compose(
        [ScaleIntensity(), EnsureChannelFirst(), Resize((224, 224, 200))]
    )

    dataset = ImageDataset(
        image_files=images,
        labels=torch.nn.functional.one_hot(torch.as_tensor(labels)).float(),
        transform=train_transforms,
    )

    return dataset


@pytest.fixture
def setup_model():
    """Fixture to setup the model and load pretrained weights."""
    root_dir = "./data_temp"
    loadmodel_path = os.path.join(root_dir, "ckpt.pth")
    state_dict = torch.load(
        loadmodel_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_different_model(model, state_dict)

    return model


def test_labels_and_features_matching(load_real_data, setup_model):
    """Test if the number of labels and features match using real data."""
    dataset = load_real_data
    model = setup_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model in evaluation mode
    model.eval()

    features_train = []
    labels_train = []

    # Simulate feature extraction (real data fed into model)
    for i in range(len(dataset)):
        image, label = dataset[i]
        image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
        label = torch.argmax(label).item()  # Get the label index

        # Forward pass through the model
        with torch.no_grad():
            _, features = model(image, return_features=True)

        features_train.append(features.cpu().numpy())
        labels_train.append(label)

    features_train = np.array(
        features_train
    ).squeeze()  # Convert list of features to array
    labels_train = np.array(labels_train)

    # Check if the number of features matches the number of labels
    assert (
        features_train.shape[0] == labels_train.shape[0]
    ), "Number of features should match the number of labels"

    # Check if the feature vector has 1024 dimensions (consistent with model output)
    assert features_train.shape[1] == 1024, "Feature vector should have 1024 dimensions"


def test_model_loading(setup_model):
    """Test if the model loads correctly with the given pretrained weights."""
    model = setup_model
    state_dict = model.state_dict()

    # Check that the model's layers are correctly initialized
    assert len(state_dict) > 0, "Model state dict should not be empty"

    # Check if specific layers exist in the model (e.g., the final layer)
    assert (
        "classification.out.weight" in state_dict
        or "features.conv0.weight" in state_dict
    ), "Model does not have the expected layer weights"


def test_real_data_loading(load_real_data):
    """Test if the real MRI data and labels are loaded correctly."""
    dataset = load_real_data

    # Check if data and labels are loaded correctly
    assert len(dataset) > 0, "The dataset should not be empty"

    for i in range(len(dataset)):
        image, label = dataset[i]
        assert image.shape == (
            1,
            224,
            224,
            200,
        ), f"Image shape should be (1, 224, 224, 200), but got {image.shape}"
        assert torch.argmax(label).item() in [0, 1], "Labels should be binary (0 or 1)"
