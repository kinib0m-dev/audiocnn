import sys
import modal
from model import AudioCNN
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import torchaudio
import torch
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Initialize Modal app for distributed computing
# Modal allows running this training script on cloud GPUs
app = modal.App("audio-cnn")

# Define the container image with all dependencies
# This image will be used to run the training job on Modal's infrastructure
image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")  # Install Python dependencies
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])  # Install system dependencies for audio processing
         .run_commands([
             # Download and extract ESC-50 dataset during image build
             "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
             "cd /tmp && unzip esc50.zip",
             "mkdir -p /opt/esc50-data",
             "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
             "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
         ])
         .add_local_python_source("model"))  # Include local model.py file in the image

# Create persistent volumes for data storage
# These volumes persist between different runs of the training job
volume = modal.Volume.from_name("esc50-data", create_if_missing=True)  # For dataset storage
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)  # For model checkpoints

class ESC50Dataset(Dataset):
    """
    Custom Dataset class for the ESC-50 environmental sound classification dataset.
    
    ESC-50 is a collection of 2000 short environmental audio recordings suitable
    for benchmarking methods of environmental sound classification. The dataset
    consists of 50 classes with 40 examples per class.
    """
    
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        """
        Initialize the ESC-50 dataset.
        
        Args:
            data_dir (str/Path): Path to the ESC-50 data directory
            metadata_file (str/Path): Path to the metadata CSV file
            split (str): Either "train" or "val" to specify dataset split
            transform: Audio transformations to apply (e.g., mel-spectrogram conversion)
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        # Split the dataset based on fold 5 (standard ESC-50 evaluation protocol)
        # Folds 1-4 are used for training, fold 5 for validation/testing
        if split == "train":
            self.metadata = self.metadata[self.metadata["fold"] != 5]
        else:
            self.metadata = self.metadata[self.metadata["fold"] == 5]

        # Get unique class names and create class-to-index mapping
        self.classes = sorted(self.metadata["category"].unique())
        
        # Create a dictionary mapping class names to numerical indices
        # This is needed for the neural network which expects integer labels
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Add numerical labels to metadata for easy access during training
        self.metadata["label"] = self.metadata["category"].map(self.class_to_idx)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (spectrogram, label) where spectrogram is the processed audio
                   and label is the class index
        """
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row["filename"]

        # Load audio file using torchaudio
        # Returns waveform tensor and sample rate
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert stereo to mono if necessary
        # Take the mean across channels to create single-channel audio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply transformations if provided (e.g., convert to mel-spectrogram)
        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            # If no transform, return raw waveform
            spectrogram = waveform

        return spectrogram, row["label"]
    
def mixup_data(x, y):
    """
    Apply mixup data augmentation technique.
    
    Mixup creates virtual training examples by mixing pairs of examples and their labels.
    This technique helps improve generalization and reduces overfitting.
    
    Args:
        x (torch.Tensor): Batch of input data
        y (torch.Tensor): Batch of labels
        
    Returns:
        tuple: (mixed_x, y_a, y_b, lam) where mixed_x is the interpolated data,
               y_a and y_b are the original labels, and lam is the mixing coefficient
    """
    # Sample mixing coefficient from Beta distribution
    # Beta(0.2, 0.2) tends to produce values near 0 or 1, creating strong mixup
    lam = np.random.beta(0.2, 0.2)

    batch_size = x.size(0)
    
    # Create random permutation of batch indices for mixing pairs
    index = torch.randperm(batch_size).to(x.device)

    # Create mixed input by linear interpolation between pairs
    # mixed_x = λ * x + (1-λ) * x_permuted
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    # Return original labels separately for mixed loss computation
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam
    
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute mixup loss for mixed training examples.
    
    The loss is a weighted combination of losses for both original labels,
    using the same mixing coefficient as the input mixing.
    
    Args:
        criterion: Loss function (e.g., CrossEntropyLoss)
        pred (torch.Tensor): Model predictions
        y_a (torch.Tensor): First set of original labels
        y_b (torch.Tensor): Second set of original labels  
        lam (float): Mixing coefficient
        
    Returns:
        torch.Tensor: Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

@app.function(
    image=image, 
    gpu="A10G",  # Use NVIDIA A10G GPU for training
    volumes={"/data": volume, "/models": model_volume},  # Mount persistent volumes
    timeout=60 * 60 * 3  # 3-hour timeout for long training runs
)
def train():
    """
    Main training function that runs on Modal's cloud infrastructure.
    
    This function implements the complete training pipeline including:
    - Data loading and preprocessing
    - Model initialization
    - Training loop with mixup augmentation
    - Validation and model checkpointing
    """
    # Setup TensorBoard
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'/models/tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir)


    # Set up data directory path
    esc50_dir = Path("/opt/esc50-data")
    
    # Define training data transformations
    # These transformations convert raw audio to mel-spectrograms and apply augmentations
    train_transform = nn.Sequential(
        # Convert audio waveform to mel-spectrogram
        T.MelSpectrogram(
            sample_rate=44100,  # Expected sample rate of ESC-50 audio
            n_fft=1024,        # FFT window size
            hop_length=512,    # Overlap between windows
            n_mels=128,        # Number of mel frequency bins
            f_min=0,           # Minimum frequency
            f_max=11025        # Maximum frequency (Nyquist frequency)
        ), 
        # Convert amplitude to decibel scale for better neural network training
        T.AmplitudeToDB(),
        # Data augmentation: randomly mask frequency bins to improve robustness
        T.FrequencyMasking(freq_mask_param=30),
        # Data augmentation: randomly mask time steps to improve robustness
        T.TimeMasking(time_mask_param=80)
    )

    # Define validation data transformations (no augmentation)
    # Only basic mel-spectrogram conversion without masking
    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=44100, 
            n_fft=1024, 
            hop_length=512, 
            n_mels=128, 
            f_min=0, 
            f_max=11025
        ), 
        T.AmplitudeToDB(),
    )

    # Create training and validation datasets
    train_dataset = ESC50Dataset(
        data_dir=esc50_dir, 
        metadata_file=esc50_dir / "meta" / "esc50.csv", 
        split="train", 
        transform=train_transform
    )

    val_dataset = ESC50Dataset(
        data_dir=esc50_dir, 
        metadata_file=esc50_dir / "meta" / "esc50.csv", 
        split="test", 
        transform=val_transform
    )

    # Print dataset statistics
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders for batched training
    # Shuffle training data for better convergence, don't shuffle validation
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Set up device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with the number of classes in the dataset
    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)  # Move model to GPU

    # Training hyperparameters
    num_epochs = 100
    
    # Loss function with label smoothing to prevent overconfident predictions
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # AdamW optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    # One-cycle learning rate scheduler for faster convergence
    # Starts low, increases to max_lr, then decreases
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,  # Peak learning rate
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1  # Spend 10% of training increasing LR, 90% decreasing
    )

    # Track best validation accuracy for model checkpointing
    best_accuracy = 0.0

    print("Starting training...")
    
    # Main training loop
    for epoch in range(num_epochs):
        # Set model to training mode (enables dropout, batch norm updates)
        model.train()
        epoch_loss = 0.0

        # Progress bar for visual feedback during training
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        # Iterate through training batches
        for data, target in progress_bar:
            # Move data to GPU
            data, target = data.to(device), target.to(device)

            # Apply mixup augmentation 30% of the time
            if np.random.random() > 0.7:
                # Apply mixup data augmentation
                data, target_a, target_b, lam = mixup_data(data, target)
                output = model(data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            else: 
                # Standard training without mixup
                output = model(data)
                loss = criterion(output, target)

            # Backpropagation and optimization step
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update model parameters
            scheduler.step()       # Update learning rate

            # Track training loss
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f'{loss.item():.4f}'})

        # Calculate average training loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        # Add graphs to the tensorboard
        writer.add_scalar("Loss/Train", avg_epoch_loss, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        # Validation phase after each epoch
        model.eval()  # Set model to evaluation mode (disables dropout, freezes batch norm)

        # Initialize validation metrics
        correct = 0
        total = 0
        val_loss = 0

        # Disable gradient computation for efficiency during validation
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                
                loss = criterion(outputs, target) 
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)  # Get class with highest probability
                total += target.size(0)
                correct += (predicted == target).sum().item()

        # Calculate validation metrics
        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(test_dataloader)
        # Add graphs to the tensorboard
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", accuracy, epoch)

        # Print epoch summary
        print(f'Epoch {epoch + 1} Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Save best model checkpoint
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save model state, accuracy, epoch, and class information
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'classes': train_dataset.classes
            }, '/models/best_model.pth') 

            print(f'New best model saved: {accuracy:.2f}%')

    # Close the writer
    writer.close()
    print(f'Training completed! Best accuracy: {best_accuracy:.2f}%')

@app.local_entrypoint()
def main():
    """
    Local entrypoint that triggers the remote training function.
    
    This function runs locally and dispatches the training job to Modal's cloud.
    """
    print(train.remote())