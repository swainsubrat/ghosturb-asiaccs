"""
Autoencoder model for surrogate model training and attacking attacks_ae
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import os

from scapy.all import *
from titli.fe import AfterImage, NetStat
from titli.utils import StreamingCSVDataset

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import logging
import argparse
from typing import Optional, Tuple, List
from joblib import dump, load as joblib_load
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, accuracy_score, roc_curve, auc)

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import glob
from pathlib import Path

from utils import plot_anomaly_scores

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PyTorchModel(nn.Module):
    def __init__(self, dataset_name: str, input_size: int, device: str):
        """
        Base class for PyTorch models with thresholding and scaler support.
        Args:
            dataset_name (str): Name of the dataset.
            input_size (int): Number of input features.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        super(PyTorchModel, self).__init__()
        # self.model_name = self.__class__.__name__
        self.dataset_name = dataset_name
        self.device = device
        self.input_size = input_size
        self.scaler = StandardScaler()
        self.epochs = 1

        self.model = self.get_model()

        self.threshold = None

    def get_model(self) -> nn.Module:
        """
        Abstract method to be overridden by specific model classes.
        Returns:
            nn.Module: The model architecture.
        """
        raise NotImplementedError("Must be implemented by the subclass")
    
    def train_model(self, train_loader, val_loader=None) -> None:
        """
        Trains the model using the provided DataLoader with validation and early stopping.
        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader, optional): DataLoader for validation data.
        """
        all_train_data = []
        for inputs, _ in train_loader:
            all_train_data.append(inputs.numpy())
        all_train_data = np.concatenate(all_train_data, axis=0)
        self.scaler.fit(all_train_data)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.train()
            running_loss = 0.0
            for inputs, _ in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                inputs = inputs.to(self.device)
                inputs_scaled = self.scaler.transform(inputs.cpu())
                inputs_scaled = np.clip(inputs_scaled, -5, 5)
                inputs_scaled = torch.tensor(inputs_scaled, dtype=torch.float32).to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self(inputs_scaled)
                loss = self.criterion(outputs, inputs_scaled)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                self.optimizer.step()
                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.6f}")

            # Validation
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, _ in val_loader:
                        inputs = inputs.to(self.device)
                        inputs_scaled = self.scaler.transform(inputs.cpu())
                        inputs_scaled = np.clip(inputs_scaled, -5, 5)
                        inputs_scaled = torch.tensor(inputs_scaled, dtype=torch.float32).to(self.device)
                        outputs = self(inputs_scaled)
                        loss = self.criterion(outputs, inputs_scaled)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                logger.info(f"Epoch {epoch + 1}, Val Loss: {avg_val_loss:.6f}")

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break


        self.calculate_threshold(train_loader)

    def save(self, model_path=None, scaler_path=None, threshold_path=None):
        """
        Saves the model weights, threshold, and scaler to disk separately.
        Args:
            model_path (str, optional): Path to save the model weights.
            scaler_path (str, optional): Path to save the scaler.
            threshold_path (str, optional): Path to save the threshold.
        """
        model_name = self.model_name.lower()
        if not model_path:
            model_path = f"../artifacts/{self.dataset_name}/models/{model_name}.pth"
        if not scaler_path:
            scaler_path = f"../artifacts/{self.dataset_name}/objects/{model_name}/scaler.joblib"
        if not threshold_path:
            threshold_path = f"../artifacts/{self.dataset_name}/objects/{model_name}/threshold.npy"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        os.makedirs(os.path.dirname(threshold_path), exist_ok=True)
        try:
            # Save only weights
            torch.save(self.state_dict(), model_path)
            # Save scaler
            dump(self.scaler, scaler_path)
            # Save threshold
            np.save(threshold_path, np.array([self.threshold]))
            logger.info(f"Model weights saved to {model_path}")
            logger.info(f"Scaler saved to {scaler_path}")
            logger.info(f"Threshold saved to {threshold_path}")
        except Exception as e:
            logger.error(f"Failed to save model/scaler/threshold: {e}")

    def load(self, model_path=None, scaler_path=None, threshold_path=None):
        """
        Loads the model weights, threshold, and scaler from disk separately.
        Args:
            model_path (str, optional): Path to load the model weights from.
            scaler_path (str, optional): Path to load the scaler from.
            threshold_path (str, optional): Path to load the threshold from.
        Returns:
            bool: True if all loaded successfully, False otherwise.
        """
        model_name = self.model_name.lower()
        if not model_path:
            model_path = f"../artifacts/{self.dataset_name}/models/{model_name}.pth"
        if not scaler_path:
            scaler_path = f"../artifacts/{self.dataset_name}/objects/{model_name}/scaler.joblib"
        if not threshold_path:
            threshold_path = f"../artifacts/{self.dataset_name}/objects/{model_name}/threshold.npy"
        try:
            # Load only weights
            self.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.scaler = joblib_load(scaler_path)
            self.threshold = float(np.load(threshold_path)[0])
            logger.info(f"Model weights loaded from {model_path}")
            logger.info(f"Scaler loaded from {scaler_path}")
            logger.info(f"Threshold loaded from {threshold_path}")
            return True
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load model/scaler/threshold: {e}")
            return False

    def calculate_threshold(self, train_loader) -> None:
        """
        Calculates the anomaly threshold based on reconstruction errors.
        Supports two methods: 'mad' (MAD in log-space) or 'std' (mean + k*std).
        Args:
            train_loader (DataLoader): DataLoader for validation/training data.
        """
        logger.info(f"Calculating threshold using {self.threshold_method.upper()} method.")
        self.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for inputs, _ in tqdm(train_loader, desc="Calculating threshold"):
                inputs = inputs.to(self.device)
                inputs_scaled = self.scaler.transform(inputs.cpu())
                inputs_scaled = np.clip(inputs_scaled, -5, 5)
                inputs_scaled = torch.tensor(inputs_scaled, dtype=torch.float32).to(self.device)
                outputs = self(inputs_scaled)
                per_sample_loss = ((outputs - inputs_scaled) ** 2).mean(dim=1).cpu().numpy()
                reconstruction_errors.extend(per_sample_loss)

        reconstruction_errors = np.array(reconstruction_errors)
        
        if self.threshold_method == 'mad':
            # MAD-based threshold in log-space (robust to outliers)
            log_errors = np.log1p(reconstruction_errors)
            median = np.median(log_errors)
            mad = np.median(np.abs(log_errors - median))
            self.threshold = np.expm1(median + self.mad_k * mad)
            logger.info(f"Threshold (MAD-based, k={self.mad_k}): {self.threshold:.6f}")
        elif self.threshold_method == 'std':
            # Traditional mean + k*std method
            mean = np.mean(reconstruction_errors)
            std = np.std(reconstruction_errors)
            self.threshold = mean + self.std_k * std
            logger.info(f"Threshold (STD-based, k={self.std_k}): {self.threshold:.6f}")
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}. Use 'mad' or 'std'.")
        
    def infer(self, test_loader, median_filter=None):
        """
        Infers on the test set and returns the true labels and predicted labels.
        Args:
            test_loader (DataLoader): DataLoader for test data.
            median_filter (int, optional): Window size for median filtering on reconstruction errors.
        Returns:
            Tuple[List[int], List[int], List[float]]: True labels, predicted labels, and reconstruction errors.
        """
        logger.info("Starting inference on the test set.")
        if self.threshold is None:
            logger.warning("Threshold not set. Please load or train before inferring.")
            return None

        logger.info(f"Using the threshold of {self.threshold:.6f}")
        self.eval()
        reconstruction_errors = []
        y_test = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Inferencing")):
                logger.debug(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
                inputs = inputs.to(self.device)
                inputs_scaled = self.scaler.transform(inputs.cpu())
                inputs_scaled = np.clip(inputs_scaled, -5, 5)
                inputs_scaled = torch.tensor(inputs_scaled, dtype=torch.float32).to(self.device)
                outputs = self(inputs_scaled)
                per_sample_error = ((outputs - inputs_scaled) ** 2).mean(dim=1).cpu().numpy()
                reconstruction_errors.extend(per_sample_error)
                y_test.extend(labels.cpu().numpy())

        # Apply median filter if specified
        if median_filter and median_filter > 1:
            from scipy.signal import medfilt
            reconstruction_errors = medfilt(reconstruction_errors, kernel_size=median_filter)
            logger.info(f"Applied median filter with kernel size {median_filter}")

        # Predict based on threshold
        y_pred = [1 if err > self.threshold else 0 for err in reconstruction_errors]

        logger.info("Inference completed.")
        logger.info(f"Total samples processed: {len(y_pred)}")
        logger.info(f"Reconstruction errors (first 5): {reconstruction_errors[:5]}")
        return y_test, y_pred, reconstruction_errors


class Autoencoder(PyTorchModel):
    def __init__(self, dataset_name: str, input_size: int, device: str, model_name: str, latent_size: int = 16, threshold_method: str = 'mad'):
        """
        Improved Autoencoder model for anomaly detection with deeper architecture.
        Args:
            dataset_name (str): Name of the dataset.
            input_size (int): Number of input features.
            device (str): Device to run the model on ('cpu' or 'cuda').
            model_name (str, optional): Model name to be saved.
            latent_size (int): Size of latent bottleneck (default: 16).
            threshold_method (str): Method for threshold calculation - 'mad' or 'std' (default: 'mad').
        """
        self.model_name = model_name if model_name else "autoencoder"
        self.latent_size = latent_size
        self.threshold_method = threshold_method
        super().__init__(dataset_name, input_size, device)

        # Deeper architecture with BatchNorm and Dropout
        self.encoder = self.get_encoder().to(self.device)
        self.decoder = self.get_decoder().to(self.device)

        self.criterion = nn.SmoothL1Loss()  # More robust to outliers
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-3)
        self.epochs = 5
        self.patience = 3
        self.mad_k = 3.0  # MAD multiplier for threshold
        self.std_k = 2.0  # Std multiplier for threshold (old method)

    def get_encoder(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.latent_size),
            nn.ReLU(),
        )

    def get_decoder(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.input_size),
        )

    def get_model(self) -> nn.Module:
        # This satisfies the base class requirement
        return nn.Sequential(
            self.get_encoder(),
            self.get_decoder()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def find_label_file(pcap_file, labels_folder):
    """
    Find the appropriate label file for a given pcap file.
    Handles cases where Adv_SYN_DoS.pcap should use SYN_DoS.csv
    
    Args:
        pcap_file (str): Path to the pcap file
        labels_folder (str): Path to the labels folder
    
    Returns:
        str or None: Path to the corresponding label file, or None if not found
    """
    pcap_name = os.path.splitext(os.path.basename(pcap_file))[0]  # e.g., "SYN_DoS" or "Adv_SYN_DoS"
    
    # First, try exact match
    exact_label_path = os.path.join(labels_folder, f"{pcap_name}.csv")
    if os.path.exists(exact_label_path):
        return exact_label_path
    
    # If no exact match and pcap starts with "Adv_", try without the "Adv_" prefix
    if pcap_name.startswith("Adv_"):
        base_name = pcap_name[4:]  # Remove "Adv_" prefix
        base_label_path = os.path.join(labels_folder, f"{base_name}.csv")
        if os.path.exists(base_label_path):
            print(f"[INFO] Using {base_name}.csv for {pcap_name}.pcap (adversarial variant)")
            return base_label_path
    
    # If still no match, search for partial matches in the labels folder
    available_labels = glob.glob(os.path.join(labels_folder, "*.csv"))
    for label_file in available_labels:
        label_name = os.path.splitext(os.path.basename(label_file))[0]
        if label_name in pcap_name or pcap_name in label_name:
            print(f"[INFO] Using {label_name}.csv for {pcap_name}.pcap (partial match)")
            return label_file
    
    print(f"[WARNING] No label file found for {pcap_name}.pcap")
    return None

def refine_attack_threshold(model, reconstruction_errors, y_test, target_recall=0.9):
    """
    Adaptively lower the threshold for attacks with poor recall.
    Only applies to single-class (all-malicious) datasets.
    
    Args:
        model: The autoencoder model with current threshold
        reconstruction_errors: Array of reconstruction errors
        y_test: True labels
        target_recall: Target recall to achieve (default: 0.9)
    
    Returns:
        tuple: (refined_threshold, y_pred_refined, metrics_dict)
    """
    from sklearn.metrics import recall_score, precision_score, f1_score
    
    y_test = np.array(y_test)
    reconstruction_errors = np.array(reconstruction_errors)
    
    # Check if this is a single-class (all-malicious) dataset
    if len(np.unique(y_test)) == 1 and y_test[0] == 1:
        # Initial prediction with current threshold
        y_pred_initial = (reconstruction_errors > model.threshold).astype(int)
        initial_recall = recall_score(y_test, y_pred_initial, zero_division=0)
        
        # If recall is already good, no need to refine
        if initial_recall >= target_recall:
            logger.info(f"Initial recall {initial_recall:.3f} already meets target {target_recall}, no refinement needed")
            return model.threshold, y_pred_initial, {
                'recall': initial_recall,
                'precision': precision_score(y_test, y_pred_initial, zero_division=0),
                'f1': f1_score(y_test, y_pred_initial, zero_division=0)
            }
        
        # Try quantile-based threshold lowering
        logger.info(f"Initial recall {initial_recall:.3f} < {target_recall}, attempting adaptive refinement...")
        
        # Find threshold that achieves target recall
        sorted_errors = np.sort(reconstruction_errors)
        target_idx = int((1 - target_recall) * len(sorted_errors))
        refined_threshold = sorted_errors[max(0, target_idx - 1)]
        
        y_pred_refined = (reconstruction_errors > refined_threshold).astype(int)
        refined_recall = recall_score(y_test, y_pred_refined, zero_division=0)
        
        logger.info(f"Threshold refined: {model.threshold:.6f} → {refined_threshold:.6f}")
        logger.info(f"Recall improved: {initial_recall:.3f} → {refined_recall:.3f}")
        
        return refined_threshold, y_pred_refined, {
            'recall': refined_recall,
            'precision': precision_score(y_test, y_pred_refined, zero_division=0),
            'f1': f1_score(y_test, y_pred_refined, zero_division=0)
        }
    
    # For mixed datasets, return original predictions
    y_pred = (reconstruction_errors > model.threshold).astype(int)
    return model.threshold, y_pred, {
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }

def process_single_pcap(pcap_file, dataset_name, device,
                        max_packets=None,
                        eval_only=False,
                        model_path=None,
                        model_name=None,
                        feature_extractor="AfterImage"):
    """
    Process a single pcap file: run inference, save scores, and evaluate if labels are available.
    
    Args:
        pcap_file (str): Path to the pcap file to process
        dataset_name (str): Name of the dataset
        device (str): Device to use (cpu or cuda)
        max_packets (int, optional): Maximum number of packets to process
        eval_only (bool): If True, skip inference and only run evaluation on existing results
        feature_extractor (str): Feature extractor name (default: AfterImage)
    
    Returns:
        dict: Results summary
    """
    attack_name = os.path.splitext(os.path.basename(pcap_file))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {attack_name}")
    print(f"{'='*60}")
    
    # Set up directory structure
    model_name = "autoencoder" if model_name is None else model_name
    metrics_dir = os.path.join("../artifacts", dataset_name, "objects", model_name, attack_name)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Define paths
    csv_path = os.path.join(metrics_dir, "anomaly_scores.csv")
    y_pred_path = os.path.join(metrics_dir, "y_pred.csv")
    
    # Extract feature cache path following data folder structure
    # Convert: ../data/{dataset}/pcaps/{subfolder}/{file}.pcap
    # To:      ../data/{dataset}/features/{feature_extractor}/{subfolder}/{file}.csv

    tail = pcap_file.split("/pcaps/")[-1]  # e.g., "malicious/SYN_DoS.pcap"
    feature_cache_path = f"../data/{dataset_name}/features/{feature_extractor}/{tail.replace('.pcap', '.csv')}"
    
    # If eval_only flag is set, skip inference and only do evaluation
    if eval_only:
        if os.path.exists(y_pred_path) and os.path.exists(csv_path):
            print(f"[INFO] Evaluation-only mode: using existing results for {attack_name}")
            try:
                # Load existing results for evaluation
                result = {
                    "pcap_file": pcap_file,
                    "attack_name": attack_name,
                    "anomaly_scores_path": csv_path,
                    "status": "success",
                    "evaluation": None,
                    "skipped_inference": True
                }
                
            except Exception as e:
                print(f"[ERROR] Failed to load existing results for evaluation: {e}")
                return {
                    "pcap_file": pcap_file,
                    "attack_name": attack_name,
                    "status": "failed",
                    "error": f"Evaluation-only mode failed: {e}"
                }
        else:
            print(f"[ERROR] Evaluation-only mode requires existing results, but files not found for {attack_name}")
            return {
                "pcap_file": pcap_file,
                "attack_name": attack_name,
                "status": "failed",
                "error": "Evaluation-only mode: no existing results found"
            }
    else:
        # Normal mode: run inference (override existing files if present)
        print(f"[INFO] Running inference for {attack_name} (will override existing results)")
        
        try:
            # Load the trained model
            model = Autoencoder(dataset_name=dataset_name, input_size=100, device=device, model_name=model_name)
            if not model.load(model_path=model_path):
                print(f"[ERROR] Failed to load trained model for {dataset_name}")
                return {
                    "pcap_file": pcap_file,
                    "attack_name": attack_name,
                    "status": "failed",
                    "error": "Failed to load trained model"
                }
            
            # Process pcap file (with feature caching)
            features = process_pcap(pcap_file, max_packets=max_packets, feature_cache_path=feature_cache_path)
            features = torch.tensor(features, dtype=torch.float32)
            
            # Run inference with dummy labels for DataLoader
            dataset = TensorDataset(features, torch.zeros(len(features)))
            loader = DataLoader(dataset, batch_size=512, shuffle=False)
            
            # Use model's infer method with median filter
            _, y_pred, reconstruction_errors = model.infer(loader, median_filter=5)
            
            # Save anomaly scores
            np.savetxt(csv_path, reconstruction_errors, delimiter=",")
            print(f"[INFO] Anomaly scores saved to {csv_path}")
            
            result = {
                "pcap_file": pcap_file,
                "attack_name": attack_name,
                "num_packets": len(reconstruction_errors),
                "threshold": model.threshold,
                "anomaly_scores_path": csv_path,
                "status": "success",
                "evaluation": None,
                "skipped_inference": False
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to process {attack_name}: {e}")
            return {
                "pcap_file": pcap_file,
                "attack_name": attack_name,
                "status": "failed",
                "error": str(e)
            }
    
    return result

def batch_inference(pcap_folder, labels_folder, dataset_name, device, 
                    max_packets=None, 
                    eval_only=False, 
                    feature_extractor="AfterImage",
                    model_name="autoencoder"):
    """
    Run inference on all pcap files in a folder.
    
    Args:
        pcap_folder (str): Path to folder containing pcap files
        labels_folder (str): Path to labels folder
        dataset_name (str): Dataset name
        device (str): Device to use (cpu or cuda)
        max_packets (int, optional): Maximum number of packets to process per file
        eval_only (bool): If True, skip inference and only run evaluation on existing results
        feature_extractor (str): Feature extractor name (default: AfterImage)
    
    Returns:
        list: List of result dictionaries
    """
    # Find all pcap files in the folder (non-recursive)
    pcap_files = glob.glob(os.path.join(pcap_folder, "*.pcap"))
    pcap_files = sorted(pcap_files)  # Sort the files
    
    if not pcap_files:
        print(f"[ERROR] No pcap files found in {pcap_folder}")
        return []
    
    print(f"[INFO] Found {len(pcap_files)} pcap files to process:")
    for i, pcap_file in enumerate(pcap_files, 1):
        print(f"  {i}. {os.path.basename(pcap_file)}")
    
    # Process each pcap file
    results = []
    for i, pcap_file in enumerate(pcap_files, 1):
        print(f"\n[INFO] Processing file {i}/{len(pcap_files)}")
        result = process_single_pcap(pcap_file, dataset_name, device, max_packets, eval_only, feature_extractor=feature_extractor)
        results.append(result)
        
        # Run evaluation for each file if labels are found
        if result["status"] == "success":
            label_file = find_label_file(pcap_file, labels_folder)
            
            if label_file:
                try:
                    print(f"[INFO] Found labels at {label_file}, running evaluation...")
                    
                    # Extract feature cache path following data folder structure
                    if "/pcaps/" in pcap_file:
                        tail = pcap_file.split("/pcaps/")[-1]
                        feature_cache_path = f"../data/{dataset_name}/features/{feature_extractor}/{tail.replace('.pcap', '.csv')}"
                    else:
                        attack_name = os.path.splitext(os.path.basename(pcap_file))[0]
                        feature_cache_path = os.path.join("../artifacts", dataset_name, "objects", "autoencoder", attack_name, "features.csv")
                    
                    # Load features for evaluation (with caching)
                    features = process_pcap(pcap_file, max_packets=max_packets, feature_cache_path=feature_cache_path)
                    features = torch.tensor(features, dtype=torch.float32)
                    
                    # Load labels
                    labels_df = pd.read_csv(label_file)
                    labels = labels_df.iloc[:, 1].values
                    
                    # Align lengths
                    min_len = min(len(labels), len(features))
                    labels = labels[:min_len]
                    features = features[:min_len]
                    
                    # Load reconstruction errors
                    reconstruction_errors = np.loadtxt(result["anomaly_scores_path"], delimiter=",")
                    reconstruction_errors = reconstruction_errors[:min_len]
                    
                    # Generate predictions
                    model = Autoencoder(dataset_name=dataset_name, input_size=100, device=device, model_name=model_name)
                    model.load()
                    y_pred = [1 if error > model.threshold else 0 for error in reconstruction_errors]
                    
                    # Save predictions
                    y_pred_path = os.path.join(os.path.dirname(result["anomaly_scores_path"]), "y_pred.csv")
                    np.savetxt(y_pred_path, y_pred, fmt='%d', delimiter=",")
                    
                    # Run evaluation
                    evaluate_and_plot(labels.tolist(), y_pred, reconstruction_errors.tolist(), 
                                    "autoencoder", result["attack_name"], model.threshold, dataset_name)
                    result["evaluation"] = "completed"
                    result["label_file"] = label_file
                    
                except Exception as e:
                    print(f"[WARNING] Evaluation failed for {result['attack_name']}: {e}")
                    result["evaluation"] = f"failed: {e}"
            else:
                result["evaluation"] = "no_labels"
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    evaluated = sum(1 for r in results if r.get("evaluation") == "completed")
    no_labels = sum(1 for r in results if r.get("evaluation") == "no_labels")
    skipped_inference = sum(1 for r in results if r.get("skipped_inference", False))
    fresh = sum(1 for r in results if r.get("skipped_inference", True) == False and r["status"] == "success")
    
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if not eval_only:
        print(f"Fresh inference: {fresh}")
    if eval_only:
        print(f"Evaluation-only mode: {skipped_inference}")
    else:
        print(f"Skipped inference (used existing): {skipped_inference}")
    print(f"Evaluated (with labels): {evaluated}")
    print(f"No labels found: {no_labels}")
    
    if failed > 0:
        print(f"\nFailed files:")
        for r in results:
            if r["status"] == "failed":
                print(f"  - {r['attack_name']}: {r.get('error', 'Unknown error')}")
    
    if skipped_inference > 0:
        print(f"\nFiles with existing results (skipped inference):")
        for r in results:
            if r.get("skipped_inference", False):
                print(f"  - {r['attack_name']}")
    
    # Save summary to file
    summary_dir = os.path.join("../artifacts", dataset_name, "objects", "autoencoder")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "batch_processing_summary.txt")
    
    with open(summary_path, "w") as f:
        f.write("Batch Processing Summary\n")
        f.write("="*50 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Mode: {'Evaluation-only' if eval_only else 'Inference + Evaluation'}\n")
        f.write(f"Total files: {len(results)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        if not eval_only:
            f.write(f"Fresh inference: {fresh}\n")
        if eval_only:
            f.write(f"Evaluation-only mode: {skipped_inference}\n")
        else:
            f.write(f"Skipped inference: {skipped_inference}\n")
        f.write(f"Evaluated: {evaluated}\n")
        f.write(f"No labels: {no_labels}\n")
        f.write("\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 30 + "\n")
        for r in results:
            f.write(f"File: {r['attack_name']}\n")
            f.write(f"  Status: {r['status']}\n")
            if r["status"] == "success":
                f.write(f"  Packets: {r.get('num_packets', 'N/A')}\n")
                f.write(f"  Evaluation: {r.get('evaluation', 'N/A')}\n")
                f.write(f"  Skipped inference: {r.get('skipped_inference', False)}\n")
                f.write(f"  Scores saved: {r.get('anomaly_scores_path', 'N/A')}\n")
            else:
                f.write(f"  Error: {r.get('error', 'Unknown')}\n")
            f.write("\n")
    
    print(f"\n[INFO] Summary saved to {summary_path}")
    return results

def process_pcap(pcap_path: str, max_packets: int = None, feature_cache_path: str = None) -> list:
    """
    Processes a single .pcap file and extracts features. Caches features to CSV for reuse.
    Args:
        pcap_path (str): Path to the .pcap file.
        max_packets (int, optional): Maximum number of packets to process. If None, processes all packets.
        feature_cache_path (str, optional): Path to save/load cached features. If None, caching is disabled.
    Returns:
        list: Extracted features.
    """
    # Check if cached features exist
    if feature_cache_path and os.path.exists(feature_cache_path):
        logger.info(f"Loading cached features from {feature_cache_path}")
        try:
            features_df = pd.read_csv(feature_cache_path, header=None, low_memory=False)
            # Convert to numeric, replacing any non-numeric values with NaN
            features_df = features_df.apply(pd.to_numeric, errors='coerce')
            # Drop rows with any NaN values
            features_df = features_df.dropna()
            feature_list = features_df.values.tolist()
            logger.info(f"Loaded {len(feature_list)} cached feature vectors")
            return feature_list
        except Exception as e:
            logger.warning(f"Failed to load cached features: {e}. Re-extracting from pcap...")
    
    # Extract features from pcap
    logger.info(f"Extracting features from {pcap_path}")
    packets = PcapReader(pcap_path)
    state = NetStat(limit=1e7)
    fe = AfterImage(file_path=pcap_path, state=state)
    feature_list = []
    
    for i, pkt in enumerate(packets):
        if i%10000 == 0:
            logger.info(f"Processing packet {i + 1}")

        if max_packets and i > max_packets:
            logger.info(f"Reached maximum packet limit of {max_packets}. Stopping processing.")
            break
            
        traffic_vector = fe.get_traffic_vector(pkt)
        if traffic_vector is None:
            continue
        features = fe.update(traffic_vector)
        features = torch.tensor(features)
        feature_list.append(features.tolist())

    logger.info(f"Processed {i} packets, extracted {len(feature_list)} feature vectors")
    
    # Save features to cache
    if feature_cache_path:
        try:
            os.makedirs(os.path.dirname(feature_cache_path), exist_ok=True)
            features_array = np.array(feature_list)
            np.savetxt(feature_cache_path, features_array, delimiter=",")
            logger.info(f"Cached features saved to {feature_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cached features: {e}")
    
    return feature_list

def evaluate_and_plot(y_test, y_pred, reconstruction_errors, model_name, pcap_base, threshold, dataset_name):
    """
    Evaluates the model on the test set, calculates evaluation metrics, and plots confusion matrix and ROC curve.
    """
    # Output paths
    metrics_dir = os.path.join("../artifacts", dataset_name, "objects", model_name, pcap_base)
    plots_dir = os.path.join("../artifacts", dataset_name, "plots", model_name)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "metrics.txt")
    cm_save_path = os.path.join(plots_dir, f"{pcap_base}_confusion_matrix.png")
    roc_save_path = os.path.join(plots_dir, f"{pcap_base}_roc.png")

    print(f"Using the threshold of {threshold:.3f}")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Basic metrics
    f1 = round(f1_score(y_test, y_pred, zero_division="warn"), 3)
    precision = round(precision_score(y_test, y_pred, zero_division="warn"), 3)
    recall = round(recall_score(y_test, y_pred, zero_division="warn"), 3)  # TPR
    accuracy = round(accuracy_score(y_test, y_pred), 3)

    # Derived metrics
    tpr = recall
    fnr = round(fn / (fn + tp), 3) if (fn + tp) else 0.0
    fpr = round(fp / (fp + tn), 3) if (fp + tn) else 0.0
    tnr = round(tn / (tn + fp), 3) if (tn + fp) else 0.0

    # Print summary
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")

    # Save metrics to file
    with open(metrics_path, "w") as file:
        file.write(f"Threshold:   {threshold:.3f}\n")
        file.write(f"Accuracy:    {accuracy:.3f}\n")
        file.write(f"Precision:   {precision:.3f}\n")
        file.write(f"Recall(TPR): {tpr:.3f}\n")
        file.write(f"F1 Score:    {f1:.3f}\n")
        file.write("\nConfusion Matrix:\n")
        file.write(f"TP: {tp}\n")
        file.write(f"TN: {tn}\n")
        file.write(f"FP: {fp}\n")
        file.write(f"FN: {fn}\n")
        file.write(f"TPR (Recall): {tpr:.3f}\n")
        file.write(f"FNR:          {fnr:.3f}\n")
        file.write(f"FPR:          {fpr:.3f}\n")
        file.write(f"TNR:          {tnr:.3f}\n")

    # Plot confusion matrix
    def fmt(x):
        return f"{x:.2f}" if x < 1e4 else f"{x:.2e}"

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="", cmap="Blues",
                xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"],
                annot_kws={"size": 12},
                cbar_kws={"format": FuncFormatter(lambda x, _: fmt(x))})
    ax = plt.gca()
    for text in ax.texts:
        text_value = float(text.get_text())
        text.set_text(fmt(text_value))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_save_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_save_path}")

    # ROC curve and EER
    if np.sum(y_test) == 0 or np.sum(y_test) == len(y_test):
        print("Warning: ROC curve cannot be computed because y_test contains only one class.")
    else:
        fpr_curve, tpr_curve, thresholds = roc_curve(y_test, reconstruction_errors)
        roc_auc = auc(fpr_curve, tpr_curve)
        eer_index = np.nanargmin(np.abs(fpr_curve - (1 - tpr_curve)))
        eer_threshold = thresholds[eer_index]
        eer = fpr_curve[eer_index]

        plt.figure(figsize=(7, 6))
        plt.plot(fpr_curve, tpr_curve, label=f"ROC Curve (AUC = {roc_auc:.3f})", color="blue")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.scatter(fpr_curve[eer_index], tpr_curve[eer_index], color='red',
                    label=f"EER = {eer:.3f} at Threshold = {eer_threshold:.3f}")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve with EER")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(roc_save_path)
        plt.close()
        print(f"ROC curve saved to {roc_save_path}")
        print(f"AUC: {roc_auc:.3f}, EER: {eer:.3f} at threshold {eer_threshold:.3f}")

        # Append AUC and EER to metrics file
        with open(metrics_path, "a") as file:
            file.write(f"\nAUC-ROC:      {roc_auc:.3f}\n")
            file.write(f"EER:          {eer:.3f}\n")
            file.write(f"EER Threshold:{eer_threshold:.3f}\n")
    
    # Plot anomaly scores over packet indices
    plot_anomaly_scores(reconstruction_errors, y_test, y_pred, threshold, model_name, pcap_base, dataset_name)

def main():
    parser = argparse.ArgumentParser(description="Surrogate Autoencoder Model CLI")
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'batch'], required=True, 
                        help="Mode: train (single file training), eval (single file evaluation), or batch (batch processing)")
    parser.add_argument('--pcap', type=str, help="Path to pcap file (required for train/eval modes)")
    parser.add_argument('--pcap_folder', type=str, help="Path to folder containing pcap files (required for batch mode)")
    parser.add_argument('--feature-extractor', type=str, default="AfterImage", help="Feature extractor to use (default: AfterImage)")
    parser.add_argument('--feature-path', type=str, default=None, help="Path to save/load feature extractor state (deprecated)")
    parser.add_argument('--labels_folder', type=str, help="Path to folder containing label files (required for batch mode)")
    parser.add_argument('--dataset', type=str, default='x-iot', help="Dataset name (default: x-iot)")
    parser.add_argument('--model_path', type=str, default=None, help="Path to save/load model")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use (cpu or cuda)")
    parser.add_argument('--labels', type=str, default=None, help="Path to test labels CSV (required for eval mode)")
    parser.add_argument('--max_packets', type=int, default=None, help="Maximum number of packets to process (default: process all)")
    parser.add_argument('--eval', action='store_true', help="Run only evaluation (skip inference if results exist)")
    parser.add_argument('--threshold-method', type=str, default='std', choices=['mad', 'std'], 
                        help="Threshold calculation method: 'mad' (MAD in log-space) or 'std' (mean+2*std, default)")
    parser.add_argument('--refine-threshold', action='store_true',
                        help="Enable adaptive threshold refinement on eval-only or eval flows")
    parser.add_argument('--target-recall', type=float, default=0.9,
                        help="Target recall to use when refining threshold (default: 0.9)")
    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == 'train':
        if not args.pcap:
            if not args.feature_path:
                print("[ERROR] --pcap or --feature-path is required for train mode")
                exit(1)
    elif args.mode == 'eval':
        if not args.pcap:
            print("[ERROR] --pcap is required for eval mode")
            exit(1)
    elif args.mode == 'batch':
        if not args.pcap_folder:
            print("[ERROR] --pcap_folder is required for batch mode")
            exit(1)
        if not args.labels_folder:
            print("[ERROR] --labels_folder is required for batch mode")
            exit(1)
        if not os.path.exists(args.pcap_folder):
            print(f"[ERROR] Pcap folder not found: {args.pcap_folder}")
            exit(1)
        if not os.path.exists(args.labels_folder):
            print(f"[ERROR] Labels folder not found: {args.labels_folder}")
            exit(1)
    model_name = args.model_path.split("/")[-1][:-4] if args.model_path else "autoencoder"

    if args.mode == 'train':
        print("[NOTE] Training must be done on benign data.")
        
        # Extract feature cache path following data folder structure
        # Convert: ../data/{dataset}/pcaps/{subfolder}/{file}.pcap
        # To:      ../data/{dataset}/features/{feature_extractor}/{subfolder}/{file}.csv
        if "/pcaps/" in args.pcap:
            tail = args.pcap.split("/pcaps/")[-1]  # e.g., "benign/benign.pcap"
            feature_cache_path = f"../data/{args.dataset}/features/{args.feature_extractor}/{tail.replace('.pcap', '.csv')}"
        else:
            # Fallback to old behavior
            pcap_base = os.path.splitext(os.path.basename(args.pcap))[0]
            feature_cache_path = os.path.join("../artifacts", args.dataset, "features", f"{pcap_base}_train.csv")
        
        features = process_pcap(args.pcap, max_packets=args.max_packets, feature_cache_path=feature_cache_path)
        features = torch.tensor(features, dtype=torch.float32)
        
        model = Autoencoder(dataset_name=args.dataset, input_size=100, device=args.device, model_name=model_name, threshold_method=args.threshold_method)
        
        # Create 90/10 train/val split
        from sklearn.model_selection import train_test_split
        train_features, val_features = train_test_split(features, test_size=0.1, random_state=42)
        
        train_dataset = TensorDataset(train_features, torch.zeros(len(train_features)))
        val_dataset = TensorDataset(val_features, torch.zeros(len(val_features)))
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        
        model.train_model(train_loader, val_loader)
        model.save(args.model_path)
        print("[INFO] Model trained and saved.")
        
    elif args.mode == 'eval':
        if not os.path.exists(args.pcap):
            print(f"[ERROR] Pcap file not found: {args.pcap}")
            exit(1)
            
        # If labels not provided, infer from pcap name
        if args.labels is None:
            pcap_path = args.pcap
            pcap_dir, pcap_file = os.path.split(pcap_path)
            pcap_base, _ = os.path.splitext(pcap_file)
            # Find the root data directory (up to /pcaps/)
            labels_path = pcap_path.replace("/pcaps/", "/labels/").replace(".pcap", ".csv")
            print(f"[INFO] Inferred labels path: {labels_path}")
            args.labels = labels_path
        
        pcap_base = os.path.splitext(os.path.basename(args.pcap))[0]
        metrics_dir = os.path.join("../artifacts", args.dataset, "objects", model_name, pcap_base)
        csv_path = os.path.join(metrics_dir, "anomaly_scores.csv")
        
        if not args.eval:
            # Mode: inference + evaluation - extract features, run inference, and evaluate
            print(f"[INFO] Running inference and evaluation on {args.pcap}")
            result = process_single_pcap(
                pcap_file=args.pcap,
                dataset_name=args.dataset,
                device=args.device,
                max_packets=args.max_packets,
                eval_only=False,
                model_name=model_name,
                feature_extractor=args.feature_extractor
            )
            
            if result["status"] == "success":
                print(f"[INFO] Inference complete. Anomaly scores saved to {csv_path}")
                
                # Now run evaluation
                labels_folder = os.path.dirname(args.labels)
                label_file = find_label_file(args.pcap, labels_folder)
                
                if label_file and os.path.exists(label_file):
                    try:
                        print(f"[INFO] Found labels at {label_file}, running evaluation...")
                        
                        # Load existing anomaly scores
                        reconstruction_errors = np.loadtxt(csv_path, delimiter=",")
                        
                        # Load labels
                        labels_df = pd.read_csv(label_file)
                        y_test = labels_df.iloc[:, 1].values
                        
                        # Align lengths
                        min_len = min(len(y_test), len(reconstruction_errors))
                        y_test = y_test[:min_len]
                        reconstruction_errors = reconstruction_errors[:min_len]
                        
                        # Load model for threshold
                        model = Autoencoder(dataset_name=args.dataset, input_size=100, device=args.device, model_name=model_name, threshold_method=args.threshold_method)
                        model.load(args.model_path)
                        
                        # Optionally apply adaptive threshold refinement
                        if args.refine_threshold:
                            refined_threshold, y_pred, metrics = refine_attack_threshold(
                                model, reconstruction_errors, y_test, target_recall=args.target_recall
                            )
                        else:
                            refined_threshold = model.threshold
                            y_pred = (reconstruction_errors > refined_threshold).astype(int)
                            metrics = None
                        
                        # Save predictions
                        y_pred_path = os.path.join(metrics_dir, "y_pred.csv")
                        np.savetxt(y_pred_path, y_pred, fmt='%d', delimiter=",")
                        print(f"[INFO] Predictions saved to {y_pred_path}")
                        
                        # Run evaluation and plotting with refined threshold
                        evaluate_and_plot(y_test.tolist(), y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred, 
                                        reconstruction_errors.tolist(), 
                                        model_name, pcap_base, refined_threshold, args.dataset)
                        result["evaluation"] = "completed"
                        
                    except Exception as e:
                        print(f"[WARNING] Evaluation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        result["evaluation"] = f"failed: {e}"
                else:
                    print(f"[WARNING] No label file found for evaluation")
                    result["evaluation"] = "no_labels"
            
        elif args.eval:
            # Mode: evaluation only - reuse existing anomaly scores
            print(f"[INFO] Running evaluation-only mode (reusing existing anomaly scores)")
            
            # Check if anomaly scores exist
            if not os.path.exists(csv_path):
                print(f"[ERROR] Anomaly scores not found at {csv_path}")
                print(f"[ERROR] Please run inference first without --eval flag")
                exit(1)
            
            labels_folder = os.path.dirname(args.labels)
            label_file = find_label_file(args.pcap, labels_folder)
            
            if not label_file or not os.path.exists(label_file):
                print(f"[ERROR] Label file not found. Cannot perform evaluation.")
                exit(1)
            
            try:
                # Load existing anomaly scores
                print(f"[INFO] Loading existing anomaly scores from {csv_path}")
                reconstruction_errors = np.loadtxt(csv_path, delimiter=",")
                
                # Load labels
                print(f"[INFO] Loading labels from {label_file}")
                labels_df = pd.read_csv(label_file)
                y_test = labels_df.iloc[:, 1].values
                
                # Align lengths
                min_len = min(len(y_test), len(reconstruction_errors))
                y_test = y_test[:min_len]
                reconstruction_errors = reconstruction_errors[:min_len]
                
                print(f"[INFO] Processing {min_len} samples")
                
                # Load model only for threshold
                print(f"[INFO] Loading model to get threshold...")
                model = Autoencoder(dataset_name=args.dataset, input_size=100, device=args.device, model_name=model_name, threshold_method=args.threshold_method)
                model.load(args.model_path)
                
                # Optionally apply adaptive threshold refinement
                if args.refine_threshold:
                    refined_threshold, y_pred, metrics = refine_attack_threshold(
                        model, reconstruction_errors, y_test, target_recall=args.target_recall
                    )
                else:
                    refined_threshold = model.threshold
                    y_pred = (reconstruction_errors > refined_threshold).astype(int)
                    metrics = None
                
                # Save predictions
                y_pred_path = os.path.join(metrics_dir, "y_pred.csv")
                np.savetxt(y_pred_path, y_pred, fmt='%d', delimiter=",")
                print(f"[INFO] Predictions saved to {y_pred_path}")
                
                # Run evaluation and plotting with refined threshold
                print(f"[INFO] Running evaluation...")
                evaluate_and_plot(y_test.tolist(), y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred, 
                                reconstruction_errors.tolist(), 
                                model_name, pcap_base, refined_threshold, args.dataset)
                
                print(f"\n[INFO] Evaluation complete!")
                
            except Exception as e:
                print(f"[ERROR] Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                exit(1)
        
        print(f"\n[INFO] Processing complete.")
        
    elif args.mode == 'batch':
        print(f"[INFO] Running batch processing on folder {args.pcap_folder}")
        results = batch_inference(
            pcap_folder=args.pcap_folder,
            labels_folder=args.labels_folder,
            dataset_name=args.dataset,
            device=args.device,
            max_packets=args.max_packets,
            eval_only=args.eval,
            feature_extractor=args.feature_extractor,
            model_name=model_name
        )
        
        successful = sum(1 for r in results if r["status"] == "success")
        print(f"\n[INFO] Batch processing complete. {successful}/{len(results)} files processed successfully.")

if __name__ == "__main__":
    main()
