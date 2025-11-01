import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
import os
import json
from tqdm import tqdm

# Adjust path to import from src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Supervised_Learning
from src.data_loader import get_pretext_loaders, get_baseline_loaders
from src.utils import setup_logging, save_pretext_plot, save_plot, find_and_save_top_k_closest

def train_pretext(model, train_loader, test_loader, optimizer, criterion, device, epochs):
    """
    Training loop for the pretext task.
    """
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc_rot': [], 'test_acc_rot': [],
        'train_acc_shear': [], 'test_acc_shear': [],
        'train_acc_color': [], 'test_acc_color': []
    }

    for epoch in range(epochs):
        model.train()
        total_loss, correct_rot, correct_shear, correct_color, total_samples = 0, 0, 0, 0, 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Pretext Train]", unit="batch")
        for images, labels in train_bar:
            images = images.to(device)
            rot_labels = labels['rotation'].to(device)
            shear_labels = labels['shear'].to(device)
            color_labels = labels['color'].to(device)

            optimizer.zero_grad()
            rot_pred, shear_pred, color_pred, _ = model(images)

            loss_rot = criterion(rot_pred, rot_labels)
            loss_shear = criterion(shear_pred, shear_labels)
            loss_color = criterion(color_pred, color_labels)
            loss = loss_rot + loss_shear + loss_color # Total loss
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct_rot += (rot_pred.argmax(1) == rot_labels).sum().item()
            correct_shear += (shear_pred.argmax(1) == shear_labels).sum().item()
            correct_color += (color_pred.argmax(1) == color_labels).sum().item()
            total_samples += images.size(0)

        history['train_loss'].append(total_loss / total_samples)
        history['train_acc_rot'].append(100 * correct_rot / total_samples)
        history['train_acc_shear'].append(100 * correct_shear / total_samples)
        history['train_acc_color'].append(100 * correct_color / total_samples)

        # Pretext Testing
        model.eval()
        total_loss, correct_rot, correct_shear, correct_color, total_samples = 0, 0, 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                rot_labels = labels['rotation'].to(device)
                shear_labels = labels['shear'].to(device)
                color_labels = labels['color'].to(device)

                rot_pred, shear_pred, color_pred, _ = model(images)
                loss_rot = criterion(rot_pred, rot_labels)
                loss_shear = criterion(shear_pred, shear_labels)
                loss_color = criterion(color_pred, color_labels)
                loss = loss_rot + loss_shear + loss_color

                total_loss += loss.item() * images.size(0)
                correct_rot += (rot_pred.argmax(1) == rot_labels).sum().item()
                correct_shear += (shear_pred.argmax(1) == shear_labels).sum().item()
                correct_color += (color_pred.argmax(1) == color_labels).sum().item()
                total_samples += images.size(0)

        history['test_loss'].append(total_loss / total_samples)
        history['test_acc_rot'].append(100 * correct_rot / total_samples)
        history['test_acc_shear'].append(100 * correct_shear / total_samples)
        history['test_acc_color'].append(100 * correct_color / total_samples)

        logging.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {history['train_loss'][-1]:.4f}, Test Loss: {history['test_loss'][-1]:.4f}")
        logging.info(f"  Train Acc - Rot: {history['train_acc_rot'][-1]:.2f}%, Shear: {history['train_acc_shear'][-1]:.2f}%, Color: {history['train_acc_color'][-1]:.2f}%")
        logging.info(f"  Test Acc  - Rot: {history['test_acc_rot'][-1]:.2f}%, Shear: {history['test_acc_shear'][-1]:.2f}%, Color: {history['test_acc_color'][-1]:.2f}%")
    
    return history

def train_finetune(model, train_loader, test_loader, optimizer, criterion, device, epochs, train_size, test_size):
    """
    Training loop for fine-tuning on the classification task.
    """
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': []
    }

    for epoch in range(epochs):
        model.train()
        train_loss, correct_train, total_train = 0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Finetune]", unit="batch")

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # We only need the classification output now
            _, _, _, class_pred = model(images)
            loss = criterion(class_pred, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            total_train += labels.size(0)
            correct_train += (class_pred.argmax(1) == labels).sum().item()

        # Finetune Testing
        model.eval()
        test_loss, correct_test, total_test = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                _, _, _, class_pred = model(images)
                loss = criterion(class_pred, labels)

                test_loss += loss.item() * images.size(0)
                total_test += labels.size(0)
                correct_test += (class_pred.argmax(1) == labels).sum().item()

        epoch_train_loss = train_loss / train_size
        epoch_train_acc = 100 * correct_train / total_train
        epoch_test_loss = test_loss / test_size
        epoch_test_acc = 100 * correct_test / total_test

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_loss'].append(epoch_test_loss)
        history['test_acc'].append(epoch_test_acc)

        logging.info(f"Epoch [{epoch+1}/{epochs}] | "
                     f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
                     f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")
    
    return history

def main(args):
    """
    Main function to run SSL pretext training and then fine-tuning.
    """
    setup_logging(log_filename="ssl_pretext.log")
    logging.info("Starting Self-Supervised (Pretext Task) Training & Fine-tuning")
    logging.info(f"Arguments: {args}")

    device = torch.device("cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = Supervised_Learning(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    
    try:
        # --- Phase 1: Pretext Task Training ---
        logging.info("--- Starting Phase 1: Pretext Task Training ---")
        pretext_train_loader, pretext_test_loader = get_pretext_loaders(args.batch_size)
        optimizer_pretext = optim.Adam(model.parameters(), lr=args.lr)
        
        pretext_history = train_pretext(
            model, pretext_train_loader, pretext_test_loader, 
            optimizer_pretext, criterion, device, args.pretext_epochs
        )
        
        logging.info("--- Pretext Training Complete ---")
        
        # Save pretext results
        model_path = os.path.join(output_dir, "ssl_pretext_model.pth")
        torch.save(model.state_dict(), model_path)
        logging.info(f"Pretext model saved to {model_path}")

        history_path = os.path.join(output_dir, "ssl_pretext_history.json")
        with open(history_path, 'w') as f:
            json.dump(pretext_history, f)
        logging.info(f"Pretext history saved to {history_path}")
        
        save_pretext_plot(pretext_history, "SSL Pretext Task Performance", "ssl_pretext_metrics.png")

        # --- Phase 2: Fine-tuning ---
        logging.info("--- Starting Phase 2: Fine-tuning on Labeled Data ---")
        finetune_train_loader, finetune_test_loader, train_size, test_size = get_baseline_loaders(
            args.batch_size, args.subset_size
        )
        
        # We can re-use the same optimizer or create a new one.
        # Let's create a new one for fine-tuning, as is common.
        optimizer_finetune = optim.Adam(model.parameters(), lr=args.lr)
        
        finetune_history = train_finetune(
            model, finetune_train_loader, finetune_test_loader,
            optimizer_finetune, criterion, device, args.finetune_epochs,
            train_size, test_size
        )
        
        logging.info("--- Fine-tuning Complete ---")
        
        # Save finetune results
        model_path = os.path.join(output_dir, "ssl_finetuned_model.pth")
        torch.save(model.state_dict(), model_path)
        logging.info(f"Finetuned model saved to {model_path}")

        history_path = os.path.join(output_dir, "ssl_finetune_history.json")
        with open(history_path, 'w') as f:
            json.dump(finetune_history, f)
        logging.info(f"Finetune history saved to {history_path}")
        
        save_plot(finetune_history, "SSL Fine-tuning Performance", "ssl_finetune_metrics.png")

        # --- Phase 3: Feature Visualization ---
        find_and_save_top_k_closest(
            model.backbone, finetune_train_loader, finetune_test_loader, 
            device, "ssl_finetune_similarity.png"
        )

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SSL Pretext Model and Finetune")
    parser.add_argument("--pretext_epochs", type=int, default=30, help="Number of pretext training epochs")
    parser.add_argument("--finetune_epochs", type=int, default=30, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--subset_size", type=int, default=5000, help="Number of labeled samples for fine-tuning")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disables CUDA training")
    
    args = parser.parse_args()
    main(args)
