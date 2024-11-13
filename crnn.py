import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from tqdm import tqdm  # For progress bars

# Constants
IMAGE_HEIGHT = 70
IMAGE_WIDTH = 250
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Define the characters that can appear in the captcha
CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for blank character (CTC requirement)


class CaptchaDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(self.image_files)} images in {folder_path}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_name = self.image_files[idx]
            img_path = os.path.join(self.folder_path, img_name)

            # Read image in grayscale
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")

            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

            # Normalize image
            image = image / 255.0
            image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension

            # Get label from filename (without extension)
            label = os.path.splitext(img_name)[0]

            # Convert label to indices
            label_indices = [CHAR_TO_IDX[c] for c in label if c in CHAR_TO_IDX]
            if not label_indices:
                raise ValueError(f"No valid characters in label: {label}")

            label_length = len(label_indices)

            return image, torch.LongTensor(label_indices), label_length

        except Exception as e:
            print(f"Error processing image {idx}: {str(e)}")
            # Return a dummy sample in case of error
            return torch.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH)), torch.LongTensor([1]), 1


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )

        self.reshape = nn.Linear(1024, 512)
        self.rnn1 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        # CNN feature extraction
        conv = self.cnn(x)
        batch, channel, height, width = conv.size()

        # Prepare for RNN
        conv = conv.permute(0, 3, 1, 2)
        conv = conv.contiguous().view(batch, width, channel * height)
        conv = self.reshape(conv)

        # RNN layers
        rnn_out, _ = self.rnn1(conv)
        rnn_out, _ = self.rnn2(rnn_out)

        # Final prediction
        output = self.fc(rnn_out)
        return output


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    model.to(device)
    print(f"Using device: {device}")

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("Training phase:")

        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f"Training")
        for batch_idx, (images, labels, label_lengths) in enumerate(train_pbar):
            try:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                batch_size, width, num_classes = outputs.size()

                # CTC loss calculation
                input_lengths = torch.full(size=(batch_size,), fill_value=width, dtype=torch.long)

                # Handle potential zero-length labels
                if torch.min(label_lengths) < 1:
                    print(f"Warning: Found zero-length label in batch {batch_idx}")
                    continue

                loss = criterion(outputs.log_softmax(2).transpose(0, 1), labels, input_lengths, label_lengths)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # Add gradient clipping
                optimizer.step()

                train_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 2)
                predicted = predicted.transpose(1, 0)

                for i in range(batch_size):
                    pred_text = ''.join([IDX_TO_CHAR[idx.item()] for idx in predicted[i] if idx.item() != 0])
                    true_text = ''.join([IDX_TO_CHAR[idx.item()] for idx in labels[i][:label_lengths[i]]])
                    if pred_text == true_text:
                        train_correct += 1
                train_total += batch_size

                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f'{train_loss / (batch_idx + 1):.4f}',
                    'acc': f'{100. * train_correct / train_total:.2f}%'
                })

            except Exception as e:
                print(f"\nError in training batch {batch_idx}: {str(e)}")
                continue

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        print("\nValidation phase:")
        val_pbar = tqdm(val_loader, desc=f"Validation")

        with torch.no_grad():
            for batch_idx, (images, labels, label_lengths) in enumerate(val_pbar):
                try:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    batch_size, width, num_classes = outputs.size()

                    # CTC loss calculation for validation
                    input_lengths = torch.full(size=(batch_size,), fill_value=width, dtype=torch.long)
                    loss = criterion(outputs.log_softmax(2).transpose(0, 1), labels, input_lengths, label_lengths)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 2)
                    predicted = predicted.transpose(1, 0)

                    for i in range(batch_size):
                        pred_text = ''.join([IDX_TO_CHAR[idx.item()] for idx in predicted[i] if idx.item() != 0])
                        true_text = ''.join([IDX_TO_CHAR[idx.item()] for idx in labels[i][:label_lengths[i]]])
                        if pred_text == true_text:
                            val_correct += 1
                    val_total += batch_size

                    # Update progress bar
                    val_pbar.set_postfix({
                        'loss': f'{val_loss / (batch_idx + 1):.4f}',
                        'acc': f'{100. * val_correct / val_total:.2f}%'
                    })

                except Exception as e:
                    print(f"\nError in validation batch {batch_idx}: {str(e)}")
                    continue

        # Print epoch statistics
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        print(f'\nEpoch {epoch + 1}/{num_epochs} Summary:')
        print(f'Training Loss: {train_loss / len(train_loader):.4f}')
        print(f'Training Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Loss: {val_loss / len(val_loader):.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        print('-' * 50)


def main():
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create datasets
        print("Loading datasets...")
        train_dataset = CaptchaDataset(r'C:\Users\rwols\Desktop\Captchas\train')
        val_dataset = CaptchaDataset(r'C:\Users\rwols\Desktop\Captchas\validation')
        test_dataset = CaptchaDataset(r'C:\Users\rwols\Desktop\Captchas\test')

        # Create dataloaders
        print("Creating dataloaders...")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize model, loss, and optimizer
        print("Initializing model...")
        model = CRNN()
        criterion = nn.CTCLoss(zero_infinity=True)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Train the model
        print("Starting training...")
        train_model(model, train_loader, val_loader, criterion, optimizer, device, NUM_EPOCHS)

        # Test phase
        print("\nStarting test phase...")
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc="Testing")
            for images, labels, label_lengths in test_pbar:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 2)
                predicted = predicted.transpose(1, 0)

                batch_size = images.size(0)
                for i in range(batch_size):
                    pred_text = ''.join([IDX_TO_CHAR[idx.item()] for idx in predicted[i] if idx.item() != 0])
                    true_text = ''.join([IDX_TO_CHAR[idx.item()] for idx in labels[i][:label_lengths[i]]])
                    if pred_text == true_text:
                        test_correct += 1
                test_total += batch_size

                # Update progress bar
                test_pbar.set_postfix({
                    'acc': f'{100. * test_correct / test_total:.2f}%'
                })

        test_accuracy = 100 * test_correct / test_total
        print(f'\nFinal Test Accuracy: {test_accuracy:.2f}%')

    except Exception as e:
        print(f"An error occurred in main: {str(e)}")


if __name__ == '__main__':
    main()