import os
import argparse
import pandas as pd
from attacks_ae import Attack
from utils import (plot_adv_traffic, calculate_aer,
                   calculate_asr, save_adversarial_metrics)

def get_args_parser():
    parser = argparse.ArgumentParser(
        "Adversarial attack using Autoencoder as the surrogate model",
        add_help=False,
    )
    parser.add_argument("--root-dir", default="../", help="folder where all the code, data, and artifacts lie")
    parser.add_argument("--pcap-path", default="../data/x-iot/pcaps/malicious/SYN_DoS.pcap", type=str)
    parser.add_argument("--benign-pcap-path", default="../data/x-iot/pcaps/benign/benign.pcap")
    parser.add_argument("--batch-size", default=1, type=int) # DON'T CHANGE THIS
    parser.add_argument("--device", default="cuda", help="device to use for training/ testing")
    parser.add_argument("--surrogate-model", default="Autoencoder", type=str, help="Name of the surrogate model")
    parser.add_argument("--dataset", default="x-iot", type=str, help="Name of the dataset")
    
    # attack params

    parser.add_argument("--attack", default="heuristics_attack", type=str, help="Name of the attack to perform or inference")
    parser.add_argument("--epsilon", default=0.0001, type=float, help="Epsilon (Maximum time perturbation) value for the attack")
    parser.add_argument("--projection-method", default="heuristics", type=str, 
                        choices=["distance", "linear", "heuristics", "cubic", "kalman", "arima"],
                        help="Projection method for time perturbation calculation")
    parser.add_argument("--spoof-ip", action="store_true", default=False, help="Weather to spoof IP or not")
    parser.add_argument("--spoof-port", action="store_true", default=False, help="Weather to spoof Port or not")
    parser.add_argument("--eval", action="store_true", default=False, help="perform attack inference")
    parser.add_argument("--threshold", default=8.6, type=float, help="Threshold of surrogate model")
    parser.add_argument("--num-features", default=100, type=int, help="Number of features used for training, 100 or 102")

    return parser


def main(args):
    if args.eval:
        # Inference mode: only load saved predictions and calculate AER
        print("Running in inference mode - loading saved predictions...")
        
        s = str(args.pcap_path).split("data/")[1].split("/")[0]
        
        # Load y_test (ground truth labels)
        labels_path = args.pcap_path.replace("/pcap/", "/labels/").replace(".pcap", ".csv")
        labels_df = pd.read_csv(labels_path)
        y_test = labels_df.iloc[:, 1].values
        print(f"Loaded y_test from: {labels_path}")
        
        # Load y_pred (original predictions)
        y_pred_path = f"../artifacts/{s}/objects/autoencoder/{args.pcap_path.split('/')[-1][:-5]}/y_pred.csv"
        y_pred = pd.read_csv(y_pred_path, header=None)[0].tolist()
        print(f"Loaded y_pred from: {y_pred_path}")
        
        # Load y_pred_adversarial (adversarial predictions)
        y_pred_adversarial_path = f"../artifacts/{s}/objects/autoencoder/{args.pcap_path.split('/')[-1][:-5]}/y_pred_adversarial.csv"
        y_pred_adversarial = pd.read_csv(y_pred_adversarial_path, header=None)[0].tolist()
        print(f"Loaded y_pred_adversarial from: {y_pred_adversarial_path}")
        
        # Calculate AER (legacy metric)
        aer = calculate_aer(y_test, y_pred, y_pred_adversarial, debug=True, stop_at_first_evasion=True)
        print(f"\nAdversarial Evasion Rate (AER):\t\t\t\t{aer:.4f} ({aer*100:.2f}%)")
        
        # Calculate ASR (comprehensive attack success metrics)
        print("\n")
        asr_results = calculate_asr(y_test, y_pred, y_pred_adversarial, verbose=True)
        print(f"\nAttack Success Rate (ASR):\t\t\t\t{asr_results['attack_success']['asr']:.4f} ({asr_results['attack_success']['asr']*100:.2f}%)")
        
        return  # Exit early for inference mode

    else:
        args.adv_pcap_path = (
            f"../data/{args.dataset}/pcaps/adversarial/{args.attack}/{args.pcap_path.split('/')[-1]}"
        )
        attack = Attack(args=args)
        
        # Map attack aliases to actual method names
        attack_aliases = {
            'ghosturb': 'heuristics_attack',
        }
        attack_method_name = attack_aliases.get(args.attack, args.attack)
        attack_method = getattr(attack, attack_method_name)
        
        result = attack_method()
        adv_packets, adv_re, total_original_time, orig_re = result
        print(f"Both original and adversarial predictions computed in parallel")

        # get original packets per second
        count_original_pkts = 0
        for pkt in adv_packets:
            if pkt[1]:
                count_original_pkts += 1

        # Print the total number of adversarial packets
        print(f"\nTotal number of adversarial malicious packets:\t\t{len(adv_packets)}")

        if len(adv_packets) > 0:
            # Calculate and print the total time taken for the adversarial attack
            total_adv_time = adv_packets[-1][0].time - adv_packets[0][0].time
            print(f"Total time required to replay the adversarial malicious packets:\t\t{total_adv_time:.4f} seconds")

            # Calculate and print the average packets per second
            avg_packets_per_second = len(adv_packets) / total_adv_time
            print(f"Average adversarial malicious packets per second:\t\t\t\t{avg_packets_per_second:.4f}")

            # Calculate and print the total number of adversarial inserted packets
            adversarial_inserted_packets = len(adv_packets) - count_original_pkts
            print(f"Total number of adversarial inserted packets:\t\t{adversarial_inserted_packets}")

        if total_original_time and total_original_time > 0:
            print("\n#############################################################################\n")

            # Print the total number of original packets
            print(f"Total number of original malicious packets:\t\t{count_original_pkts}")

            print(f"Total time required to replay the original packets:\t\t{total_original_time:.4f} seconds")

            # Calculate and print the average original packets per second
            avg_original_packets_per_second = count_original_pkts / total_original_time
            print(f"Average original packets per second:\t\t\t{avg_original_packets_per_second:.4f}")

        # get anomaly score for the adversarial packets
        if not adv_re:
            raise ValueError("No adversarial reconstruction errors found")

        # calculate adversarial evasion rate (not replay evasion rate)
        y_pred_adversarial = [1 if re > args.threshold else 0 for re in adv_re]
        
        # Save adversarial anomaly scores (reconstruction errors)
        anomaly_scores_adversarial_path = f"../artifacts/{args.dataset}/objects/autoencoder/{args.pcap_path.split('/')[-1][:-5]}/anomaly_scores_adversarial.csv"
        os.makedirs(os.path.dirname(anomaly_scores_adversarial_path), exist_ok=True)
        pd.DataFrame(adv_re).to_csv(anomaly_scores_adversarial_path, header=False, index=False)
        print(f"Saved anomaly_scores_adversarial to: {anomaly_scores_adversarial_path}")
        
        # Use original predictions computed in parallel if available, otherwise load from CSV
        if orig_re is not None:
            print("Using original predictions computed in parallel")
            y_pred = [1 if re > args.threshold else 0 for re in orig_re]
            
            # Save both predictions for future inference
            y_pred_path = f"../artifacts/{args.dataset}/objects/autoencoder/{args.pcap_path.split('/')[-1][:-5]}/y_pred.csv"
            y_pred_adversarial_path = f"../artifacts/{args.dataset}/objects/autoencoder/{args.pcap_path.split('/')[-1][:-5]}/y_pred_adversarial.csv"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(y_pred_path), exist_ok=True)
            
            pd.DataFrame(y_pred).to_csv(y_pred_path, header=False, index=False)
            pd.DataFrame(y_pred_adversarial).to_csv(y_pred_adversarial_path, header=False, index=False)
            print(f"Saved y_pred to: {y_pred_path}")
            print(f"Saved y_pred_adversarial to: {y_pred_adversarial_path}")
        else:
            print("Loading original predictions from CSV file")
            y_pred = pd.read_csv(f"../artifacts/{args.dataset}/objects/autoencoder/{args.pcap_path.split('/')[-1][:-5]}/y_pred.csv", header=None)[0].tolist()
            
            # Save y_pred_adversarial at the same location as y_pred
            y_pred_adversarial_path = f"../artifacts/{args.dataset}/objects/autoencoder/{args.pcap_path.split('/')[-1][:-5]}/y_pred_adversarial.csv"
            pd.DataFrame(y_pred_adversarial).to_csv(y_pred_adversarial_path, header=False, index=False)
            print(f"Saved y_pred_adversarial to: {y_pred_adversarial_path}")
        
        labels_path = args.pcap_path.replace("/pcaps/", "/labels/").replace(".pcap", ".csv")
        labels_df = pd.read_csv(labels_path)
        y_test = labels_df.iloc[:, 1].values
        
        # Ensure all arrays have the same length
        min_length = min(len(y_test), len(y_pred), len(y_pred_adversarial))
        y_test = y_test[:min_length]
        y_pred = y_pred[:min_length]
        y_pred_adversarial = y_pred_adversarial[:min_length]
        
        print(f"Computing AER with {min_length} samples")
        aer = calculate_aer(y_test, y_pred, y_pred_adversarial)
        print(f"\nAdversarial Evasion Rate (AER):\t\t\t\t{aer:.4f} ({aer*100:.2f}%)")
        
        # Calculate ASR (comprehensive attack success metrics)
        print("\n")
        asr_results = calculate_asr(y_test, y_pred, y_pred_adversarial, verbose=True)
        print(f"\nAttack Success Rate (ASR):\t\t\t\t{asr_results['attack_success']['asr']:.4f} ({asr_results['attack_success']['asr']*100:.2f}%)")

        # Save metrics_adversarial.txt in structured format
        metrics_adv_path = f"../artifacts/{args.dataset}/objects/autoencoder/{args.pcap_path.split('/')[-1][:-5]}/metrics_adversarial.txt"
        save_adversarial_metrics(asr_results, aer, y_test, metrics_adv_path)

        plot_adv_traffic(adv_re, adv_packets, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Adversarial Attack on Kitsune IDS", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
