import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from scapy.all import *
from titli.fe import AfterImage, NetStat

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def calculate_aer(y_test, y_pred, y_pred_adversarial, debug=False, stop_at_first_evasion=False):
    """
    Calculate Adversarial Evasion Rate using the formula: 1 - (TP_adv / TP_mal)
    
    Where:
    - TP_mal: True Positives in original predictions (y_test==1 AND y_pred==1)
    - TP_adv: True Positives in adversarial predictions (y_test==1 AND y_pred_adversarial==1)
    - AER = 1 - (TP_adv / TP_mal) = fraction of true positives that evaded detection
    """
    if len(y_test) != len(y_pred) or len(y_pred) != len(y_pred_adversarial):
        min_length = min(len(y_test), len(y_pred), len(y_pred_adversarial))
        y_test = y_test[:min_length]
        y_pred = y_pred[:min_length]
        y_pred_adversarial = y_pred_adversarial[:min_length]

    # TP_mal: True Positives in original predictions (actually malicious AND predicted malicious)
    TP_mal = sum(1 for y_true, y_pred_val in zip(y_test, y_pred) if y_true == 1 and y_pred_val == 1)
    
    # TP_adv: True Positives in adversarial predictions (actually malicious AND still predicted malicious)
    TP_adv = sum(1 for y_true, y_adv in zip(y_test, y_pred_adversarial) if y_true == 1 and y_adv == 1)

    if TP_mal == 0:
        return 0.0

    # AER = 1 - (TP_adv / TP_mal)
    return 1 - (TP_adv / TP_mal)

def calculate_asr(y_test, y_pred, y_pred_adversarial, verbose=False):
    """
    Calculate Attack Success Rate (ASR) with comprehensive classification metrics.
    
    ASR = FN_adv / TP_orig
    - Measures what percentage of originally detected attacks successfully evaded detection
    - Also computes confusion matrix, accuracy, precision, recall, F1 for both original and adversarial
    
    Returns:
        dict: Comprehensive metrics including:
            - original: {confusion_matrix, accuracy, precision, recall, f1_score}
            - adversarial: {confusion_matrix, accuracy, precision, recall, f1_score}
            - degradation: {accuracy, precision, recall, f1_score deltas}
            - attack_success: {tp_orig_count, fn_adv_from_tp_count, tp_still_detected, asr}
    """
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    
    # Ensure arrays are same length
    if len(y_test) != len(y_pred) or len(y_pred) != len(y_pred_adversarial):
        min_len = min(len(y_test), len(y_pred), len(y_pred_adversarial))
        y_test = y_test[:min_len]
        y_pred = y_pred[:min_len]
        y_pred_adversarial = y_pred_adversarial[:min_len]
    
    # Convert to numpy arrays if needed
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_pred_adversarial = np.array(y_pred_adversarial)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(y_test) | np.isnan(y_pred) | np.isnan(y_pred_adversarial))
    y_test = y_test[valid_mask]
    y_pred = y_pred[valid_mask]
    y_pred_adversarial = y_pred_adversarial[valid_mask]
    
    if len(y_test) == 0:
        if verbose:
            print("No valid samples after removing NaN values")
        return {
            'original': {'confusion_matrix': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}, 
                        'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
            'adversarial': {'confusion_matrix': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}, 
                           'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
            'degradation': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
            'attack_success': {'tp_orig_count': 0, 'fn_adv_from_tp_count': 0, 'tp_still_detected': 0, 'asr': 0.0}
        }
    
    # Calculate confusion matrices with explicit labels to ensure 2x2 matrix
    cm_orig = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_adv = confusion_matrix(y_test, y_pred_adversarial, labels=[0, 1])
    
    # Extract values from confusion matrices
    tn_orig, fp_orig, fn_orig, tp_orig_cm = cm_orig.ravel()
    tn_adv, fp_adv, fn_adv, tp_adv = cm_adv.ravel()
    
    # Calculate standard metrics for original predictions
    acc_orig = accuracy_score(y_test, y_pred)
    prec_orig = precision_score(y_test, y_pred, zero_division=0)
    rec_orig = recall_score(y_test, y_pred, zero_division=0)
    f1_orig = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate standard metrics for adversarial predictions
    acc_adv = accuracy_score(y_test, y_pred_adversarial)
    prec_adv = precision_score(y_test, y_pred_adversarial, zero_division=0)
    rec_adv = recall_score(y_test, y_pred_adversarial, zero_division=0)
    f1_adv = f1_score(y_test, y_pred_adversarial, zero_division=0)
    
    # Calculate Attack Success Rate (ASR)
    # Find TPs that became FNs after adversarial perturbation
    tp_orig_mask = (y_test == 1) & (y_pred == 1)
    fn_adv_from_tp_mask = tp_orig_mask & (y_pred_adversarial == 0)
    
    tp_orig_count = np.sum(tp_orig_mask)
    fn_adv_from_tp_count = np.sum(fn_adv_from_tp_mask)
    tp_still_detected = tp_orig_count - fn_adv_from_tp_count
    
    asr = fn_adv_from_tp_count / tp_orig_count if tp_orig_count > 0 else 0.0
    
    if verbose:
        print("=" * 80)
        print(" " * 30 + "ATTACK SUCCESS RATE (ASR)")
        print("=" * 80)
        
        print(f"\n Dataset Statistics:")
        print(f"   Total samples: {len(y_test)}")
        print(f"   Actual malicious (y_test=1): {np.sum(y_test == 1)}")
        print(f"   Actual benign (y_test=0): {np.sum(y_test == 0)}")
        
        # Original Performance
        print(f"\n{'='*80}")
        print(f" ORIGINAL DETECTION PERFORMANCE")
        print(f"{'='*80}")
        print(f"\n   Confusion Matrix:")
        print(f"                    Predicted")
        print(f"                Benign  Malicious")
        print(f"   Actual  Benign    {tn_orig:6d}     {fp_orig:6d}   (TN, FP)")
        print(f"           Malicious {fn_orig:6d}     {tp_orig_cm:6d}   (FN, TP)")
        print(f"\n   Metrics:")
        print(f"   • Accuracy:  {acc_orig:.4f} ({acc_orig*100:.2f}%)")
        print(f"   • Precision: {prec_orig:.4f} ({prec_orig*100:.2f}%)")
        print(f"   • Recall:    {rec_orig:.4f} ({rec_orig*100:.2f}%)")
        print(f"   • F1-Score:  {f1_orig:.4f}")
        
        # Adversarial Performance
        print(f"\n{'='*80}")
        print(f"  ADVERSARIAL DETECTION PERFORMANCE")
        print(f"{'='*80}")
        print(f"\n   Confusion Matrix:")
        print(f"                    Predicted")
        print(f"                Benign  Malicious")
        print(f"   Actual  Benign    {tn_adv:6d}     {fp_adv:6d}   (TN, FP)")
        print(f"           Malicious {fn_adv:6d}     {tp_adv:6d}   (FN, TP)")
        print(f"\n   Metrics:")
        print(f"   • Accuracy:  {acc_adv:.4f} ({acc_adv*100:.2f}%)")
        print(f"   • Precision: {prec_adv:.4f} ({prec_adv*100:.2f}%)")
        print(f"   • Recall:    {rec_adv:.4f} ({rec_adv*100:.2f}%)")
        print(f"   • F1-Score:  {f1_adv:.4f}")
        
        # Performance Degradation
        print(f"\n{'='*80}")
        print(f" PERFORMANCE DEGRADATION")
        print(f"{'='*80}")
        print(f"\n   Accuracy:  {acc_orig:.4f} → {acc_adv:.4f}  (Δ = {acc_orig - acc_adv:+.4f})")
        print(f"   Precision: {prec_orig:.4f} → {prec_adv:.4f}  (Δ = {prec_orig - prec_adv:+.4f})")
        print(f"   Recall:    {rec_orig:.4f} → {rec_adv:.4f}  (Δ = {rec_orig - rec_adv:+.4f})")
        print(f"   F1-Score:  {f1_orig:.4f} → {f1_adv:.4f}  (Δ = {f1_orig - f1_adv:+.4f})")
        
        # ASR Analysis
        print(f"\n{'='*80}")
        print(f" ATTACK SUCCESS RATE (ASR)")
        print(f"{'='*80}")
        print(f"\n   True Positives in original: {tp_orig_count}")
        print(f"   TP evaded (became FN): {fn_adv_from_tp_count}")
        print(f"   TP still detected: {tp_still_detected}")
        print(f"   Attack Success Rate: {asr * 100:.2f}%")
        print(f"\n   ╔{'═' * 76}╗")
        print(f"   ║  ASR = FN_adv / TP_orig = {fn_adv_from_tp_count} / {tp_orig_count} = {asr:.4f}  {' '*(76-len(str(fn_adv_from_tp_count))-len(str(tp_orig_count))-31)}║")
        print(f"   ╚{'═' * 76}╝")
        print("=" * 80)
    
    return {
        'original': {
            'confusion_matrix': {'tn': int(tn_orig), 'fp': int(fp_orig), 'fn': int(fn_orig), 'tp': int(tp_orig_cm)},
            'accuracy': float(acc_orig),
            'precision': float(prec_orig),
            'recall': float(rec_orig),
            'f1_score': float(f1_orig),
        },
        'adversarial': {
            'confusion_matrix': {'tn': int(tn_adv), 'fp': int(fp_adv), 'fn': int(fn_adv), 'tp': int(tp_adv)},
            'accuracy': float(acc_adv),
            'precision': float(prec_adv),
            'recall': float(rec_adv),
            'f1_score': float(f1_adv),
        },
        'degradation': {
            'accuracy': float(acc_orig - acc_adv),
            'precision': float(prec_orig - prec_adv),
            'recall': float(rec_orig - rec_adv),
            'f1_score': float(f1_orig - f1_adv),
        },
        'attack_success': {
            'tp_orig_count': int(tp_orig_count),
            'fn_adv_from_tp_count': int(fn_adv_from_tp_count),
            'tp_still_detected': int(tp_still_detected),
            'asr': float(asr),
        }
    }

def save_adversarial_metrics(asr_results, aer, y_test, output_path):
    """
    Save adversarial attack metrics to a structured text file for easy extraction.
    
    Args:
        asr_results: Dictionary containing ASR calculation results from calculate_asr()
        aer: Adversarial Evasion Rate (float)
        y_test: Ground truth labels (numpy array or list)
        output_path: Path where metrics_adversarial.txt will be saved
    """
    import os
    import numpy as np
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Dataset statistics
        f.write(f"total_samples: {len(y_test)}\n")
        f.write(f"actual_malicious: {np.sum(y_test == 1)}\n")
        f.write(f"actual_benign: {np.sum(y_test == 0)}\n")
        f.write("\n")
        
        # Original detection performance
        f.write("# ORIGINAL DETECTION PERFORMANCE\n")
        f.write(f"original_tn: {asr_results['original']['confusion_matrix']['tn']}\n")
        f.write(f"original_fp: {asr_results['original']['confusion_matrix']['fp']}\n")
        f.write(f"original_fn: {asr_results['original']['confusion_matrix']['fn']}\n")
        f.write(f"original_tp: {asr_results['original']['confusion_matrix']['tp']}\n")
        f.write(f"original_accuracy: {asr_results['original']['accuracy']:.6f}\n")
        f.write(f"original_precision: {asr_results['original']['precision']:.6f}\n")
        f.write(f"original_recall: {asr_results['original']['recall']:.6f}\n")
        f.write(f"original_f1_score: {asr_results['original']['f1_score']:.6f}\n")
        f.write("\n")
        
        # Adversarial detection performance
        f.write("# ADVERSARIAL DETECTION PERFORMANCE\n")
        f.write(f"adversarial_tn: {asr_results['adversarial']['confusion_matrix']['tn']}\n")
        f.write(f"adversarial_fp: {asr_results['adversarial']['confusion_matrix']['fp']}\n")
        f.write(f"adversarial_fn: {asr_results['adversarial']['confusion_matrix']['fn']}\n")
        f.write(f"adversarial_tp: {asr_results['adversarial']['confusion_matrix']['tp']}\n")
        f.write(f"adversarial_accuracy: {asr_results['adversarial']['accuracy']:.6f}\n")
        f.write(f"adversarial_precision: {asr_results['adversarial']['precision']:.6f}\n")
        f.write(f"adversarial_recall: {asr_results['adversarial']['recall']:.6f}\n")
        f.write(f"adversarial_f1_score: {asr_results['adversarial']['f1_score']:.6f}\n")
        f.write("\n")
        
        # Performance degradation
        f.write("# PERFORMANCE DEGRADATION\n")
        f.write(f"degradation_accuracy: {asr_results['degradation']['accuracy']:.6f}\n")
        f.write(f"degradation_precision: {asr_results['degradation']['precision']:.6f}\n")
        f.write(f"degradation_recall: {asr_results['degradation']['recall']:.6f}\n")
        f.write(f"degradation_f1_score: {asr_results['degradation']['f1_score']:.6f}\n")
        f.write("\n")
        
        # Attack success metrics
        f.write("# ATTACK SUCCESS METRICS\n")
        f.write(f"tp_orig_count: {asr_results['attack_success']['tp_orig_count']}\n")
        f.write(f"fn_adv_from_tp_count: {asr_results['attack_success']['fn_adv_from_tp_count']}\n")
        f.write(f"tp_still_detected: {asr_results['attack_success']['tp_still_detected']}\n")
        f.write(f"asr: {asr_results['attack_success']['asr']:.6f}\n")
        f.write(f"aer: {aer:.6f}\n")
    
    print(f"Saved metrics_adversarial.txt to: {output_path}")

def plot_anomaly_scores(reconstruction_errors, y_test, y_pred, threshold, model_name, pcap_base, dataset_name):
    """
    Plot anomaly scores (reconstruction errors) over packet indices with predictions.
    Similar to plot_adv_traffic but for evaluation of surrogate models.
    
    Args:
        reconstruction_errors: Array of anomaly scores (reconstruction errors)
        y_test: Ground truth labels
        y_pred: Predicted labels
        threshold: Detection threshold
        model_name: Name of the model (e.g., 'autoencoder')
        pcap_base: Base name of the pcap file
        dataset_name: Dataset name
    """
    import os
    
    plt.figure(figsize=(12, 8))
    
    # Generate indices for the x-axis
    packet_indices = np.arange(len(reconstruction_errors))
    
    # Define color mapping based on predictions: 0 (benign) -> green, 1 (malicious) -> red
    color_mapping = {0: 'green', 1: 'red'}
    colors = [color_mapping[pred] for pred in y_pred]
    
    # Plot the anomaly scores
    plt.scatter(
        packet_indices,
        reconstruction_errors,
        c=colors,
        alpha=0.6,
        s=1.5,
        label="Anomaly Scores (Green=Benign, Red=Malicious)"
    )
    
    # Plot the threshold line
    plt.axhline(y=threshold, color="blue", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.3f})")
    
    # Set the title and labels
    plt.title(f"Anomaly Scores - {pcap_base}", fontsize=20, fontweight='bold')
    plt.xlabel("Packet Index", fontsize=15, fontweight='bold')
    plt.ylabel("Anomaly Score (Reconstruction Error)", fontsize=15, fontweight='bold')
    
    # Set the y-axis to log scale
    plt.yscale("log")
    
    # Increase tick size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(fontsize=14, loc='upper right')
    
    # Use tight layout to prevent overlap
    plt.tight_layout()
    
    # Define output path
    plots_dir = os.path.join("../artifacts", dataset_name, "plots", model_name)
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{pcap_base}_anomaly_scores.png")
    
    # Save the plot
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Anomaly scores plot saved to {plot_path}")

def plot_adv_traffic(adv_re, adv_packets, args):
    plt.figure(figsize=(12, 8))

    # Extract and format the title
    titles = args.pcap_path.split('/')[-1][:-5].split('_')
    title = f"{' '.join(titles[:2])} attack on {' '.join(titles[-1].split('-'))}"

    # Generate indices for the x-axis
    packet_indices = np.arange(len(adv_re))

    # Define color mapping: 0 -> violet, 1 -> red
    color_mapping = {0: 'black', 1: 'red'}
    colors = [color_mapping[adv_packet[1]] for adv_packet in adv_packets]

    # Plot the adversarial malicious data
    plt.scatter(
        packet_indices,
        adv_re,
        c=colors,
        label="Adversarial Malicious",
        alpha=1,
        s=1.5
    )

    # Plot the threshold line
    plt.axhline(y=args.threshold, color="blue", linestyle="--", label="Threshold")

    # Set the title and labels with appropriate font sizes and bold font
    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel("Packet Index", fontsize=15, fontweight='bold')
    plt.ylabel("Anomaly Score", fontsize=15, fontweight='bold')

    # Set the y-axis to log scale
    plt.yscale("log")

    # Increase tick size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add legend with increased font size and position it outside the plot area
    plt.legend(fontsize=14, loc='upper right')

    # Use tight layout to prevent overlap
    plt.tight_layout()

    # Define the folder path
    s=str(args.pcap_path).split("data/")[1].split("/")[0]

    plot_path = f"../artifacts/plots/{args.attack}/{s}/{args.pcap_path.split('/')[-1][:-5]}_{args.attack}.png"
    folder_path = os.path.dirname(plot_path)

    # Check if the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Show or save the plot
    plt.savefig(plot_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    pass