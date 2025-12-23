import os
import time
import torch
import traceback
import numpy as np
import pandas as pd
import torch.nn as nn

from scapy.all import *
from scapy.layers.inet import IP
from collections import deque

from engine import *
from titli.fe import AfterImage, NetStat

from surrogate import Autoencoder
# from surrogate_pre_batchnorm import Autoencoder as AutoencoderPBN

from scapy.all import sendp

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

def denormalize_packet_size(normalized_size):
    min_size = 64  # minimum frame size
    max_size = 1518  # maximum frame size

    packet_size = normalized_size * (max_size - min_size) + min_size
    return packet_size

class Attack:
    def __init__(self, args):
        self.model = Autoencoder(dataset_name=args.dataset, input_size=args.num_features, model_name="autoencoder", device=args.device)
        self.model.load(model_path=f"../artifacts/{args.dataset}/models/autoencoder.pth")
        
        self.batch_size = args.batch_size
        self.pcap_path = args.pcap_path
        self.benign_pcap_path = args.benign_pcap_path
        self.device = args.device
        self.model.eval()

        self.args = args
        self.epsilon = args.epsilon
        self.threshold = args.threshold
        self.benign_pcap_path = args.benign_pcap_path
        self.adv_pcap_path = args.adv_pcap_path
        self.spoof_ip = args.spoof_ip
        self.spoof_port = args.spoof_port
        self.projection_method = getattr(args, 'projection_method', 'heuristics')  # Default to heuristics if not specified

    def heuristics_attack(self):

        # the attack is designed for one packet at a time
        # self.criterion = RMSELoss().to(self.device)
        self.criterion = nn.SmoothL1Loss().to(self.device)
        print(f"Attacking {self.pcap_path.split('/')[-1]} of dataset {self.args.dataset}")
        print(f"Using projection method: {self.projection_method.upper()}")

        # Load labels from CSV file
        labels_path = self.pcap_path.replace("/pcaps/", "/labels/").replace(".pcap", ".csv")
        print(f"Loading labels from: {labels_path}")
        labels_df = pd.read_csv(labels_path)
        labels = labels_df.iloc[:, 1].values  # Assuming label is in second column (index 1)
        print(f"Loaded {len(labels)} labels")

        # load the dataset
        packets = PcapReader(self.pcap_path)
        adversarial_packets = []

        # define packet parser and feature extractor for adversarial processing
        state_adv = NetStat(limit=1e8)
        fe_adv = AfterImage(file_path=None, state=state_adv)
        
        # define packet parser and feature extractor for original processing (parallel)
        state_orig = NetStat(limit=1e8)
        fe_orig = AfterImage(file_path=None, state=state_orig)

        # define constants
        adv_re = []
        orig_re = []  # Store original predictions for comparison
        carry = 0
        queue = deque(maxlen=10)
        start = time.time()

        for i, pkt in enumerate(packets):

            if i == 0:
                original_start_time = pkt.time
            else:
                original_end_time = pkt.time
            
            # Check if we've exceeded the label array bounds
            if i >= len(labels):
                break
            
            # Check label instead of IP filtering
            if labels[i] == 0:
                # Process as benign packet - compute original prediction only
                data = {"timestamp": pkt.time + carry}
                traffic_vector_orig = fe_orig.get_traffic_vector(pkt)
                traffic_vector_adv = fe_adv.get_traffic_vector(pkt)
                
                if traffic_vector_adv is None:
                    continue
                else:
                    traffic_vector_adv[-2] = traffic_vector_adv[-2] + carry

                # Original prediction (no perturbation)
                features_orig = fe_orig.update(traffic_vector_orig)
                features_orig = torch.tensor(self.model.scaler.transform(
                    features_orig.reshape(1, -1)), dtype=torch.float32).to(self.device)
                outputs_orig = self.model(features_orig)
                anomaly_score_orig = self.criterion(outputs_orig, features_orig)
                
                # Adversarial prediction (same as original for benign packets)
                features_adv = fe_adv.update(traffic_vector_adv)
                features_adv = torch.tensor(self.model.scaler.transform(
                    features_adv.reshape(1, -1)), dtype=torch.float32).to(self.device)
                outputs_adv = self.model(features_adv)
                anomaly_score_adv = self.criterion(outputs_adv, features_adv)
                
                data["anomaly_score"] = float(anomaly_score_adv.item())
                queue.append(data)
                adversarial_packets.append((pkt, 1))
                
                # Store both predictions
                orig_re.append(anomaly_score_orig.detach().cpu().numpy())
                adv_re.append(anomaly_score_adv.detach().cpu().numpy())
                
                if i % 1000 == 0:
                    print(f"Benign Packet: {i+1}, Orig Score: {anomaly_score_orig.data:.6f}, Adv Score: {anomaly_score_adv.data:.6f}")
                continue

            if i > 100000:
                break

            try:
                data = {"timestamp": pkt.time + carry}
                
                # step 1: extract parsed packet values for both streams
                traffic_vector_orig = fe_orig.get_traffic_vector(pkt)
                traffic_vector_adv = fe_adv.get_traffic_vector(pkt)

                if i < 2:
                    # Original prediction (no perturbation)
                    features_orig = fe_orig.update(traffic_vector_orig)
                    features_orig = torch.tensor(self.model.scaler.transform(
                        features_orig.reshape(1, -1)), dtype=torch.float32).to(self.device)
                    outputs_orig = self.model(features_orig)
                    anomaly_score_orig = self.criterion(outputs_orig, features_orig)
                    
                    # Adversarial prediction (same as original for first 2 packets)
                    features_adv = fe_adv.update(traffic_vector_adv)
                    features_adv = torch.tensor(self.model.scaler.transform(
                        features_adv.reshape(1, -1)), dtype=torch.float32).to(self.device)
                    outputs_adv = self.model(features_adv)
                    anomaly_score_adv = self.criterion(outputs_adv, features_adv)
                    
                    data["anomaly_score"] = float(anomaly_score_adv.item())
                    queue.append(data)
                    adversarial_packets.append((pkt, 1))
                    
                    # Store both predictions
                    orig_re.append(anomaly_score_orig.detach().cpu().numpy())
                    adv_re.append(anomaly_score_adv.detach().cpu().numpy())
                    continue

                # ORIGINAL PREDICTION (No perturbation) - Process first
                if traffic_vector_orig is not None:
                    features_orig = fe_orig.update(traffic_vector_orig)
                    features_orig = torch.tensor(self.model.scaler.transform(
                        features_orig.reshape(1, -1)), dtype=torch.float32).to(self.device)
                    outputs_orig = self.model(features_orig)
                    anomaly_score_orig = self.criterion(outputs_orig, features_orig)
                    orig_re.append(anomaly_score_orig.detach().cpu().numpy())
                else:
                    # If traffic vector is None, append NaN or skip
                    orig_re.append(np.nan)
                    continue

                # ADVERSARIAL PREDICTION (With perturbation) - Process second
                # step 2: add carry value from previous iterations
                # traffic_vector[-2]: timestamp
                if traffic_vector_adv is not None:
                    traffic_vector_adv[-2] = traffic_vector_adv[-2] + carry
                else:
                    adv_re.append(np.nan)
                    continue

                # step 2.5: get data for generating perturbations
                fake_features = fe_adv.peek([traffic_vector_adv])
                fake_features =  torch.tensor(self.model.scaler.transform(
                    np.array(fake_features).reshape(1, -1)), dtype=torch.float32).to(self.device)
                outputs = self.model(fake_features)
                fake_anomaly_score = self.criterion(outputs, fake_features)
                data["anomaly_score"] = float(fake_anomaly_score.item())
                # if i % 1000 == 0:
                #     print(f"Fake Anomaly Score: {fake_anomaly_score.item()}")

                queue.append(data)

                # step 3: generate perturbations using selected projection method
                time_perturbation = get_perturbation(
                    queue, self.threshold, self.epsilon, method=self.projection_method
                )
                
                # pop the last element from the queue to insert the updated one or new packets
                queue.pop()

                if self.spoof_port:
                    pkt = change_sport(pkt)
                # if self.spoof_ip and time_perturbation == self.epsilon:
                if self.spoof_ip:
                    pkt = change_ip(pkt)
                #step 4: apply perturbation and update the carry
                pkt.time = pkt.time + carry + time_perturbation
                # malicious_packet_queue.append(pkt)
                traffic_vector_adv = fe_adv.get_traffic_vector(pkt)

                carry = carry + time_perturbation

                # step 5: extract updated feature for adversarial prediction
                features_adv = fe_adv.update(traffic_vector_adv)
                features_adv = torch.tensor(self.model.scaler.transform(
                    features_adv.reshape(1, -1)), dtype=torch.float32).to(self.device)
                outputs_adv = self.model(features_adv)
                adv_loss = self.criterion(outputs_adv, features_adv)
                anomaly_score_adv = self.criterion(outputs_adv, features_adv)

                data = {"timestamp": pkt.time}
                data["anomaly_score"] = float(anomaly_score_adv.item())
                queue.append(data)

                if i % 1000 == 0:
                    print(f"Malicious Packet: {i+1}, Orig Score: {anomaly_score_orig.data:.6f}, Adv Score: {anomaly_score_adv.data:.6f}")

                adv_re.append(adv_loss.detach().cpu().numpy())
                adversarial_packets.append((pkt, 1))

            except Exception as e:
                print(traceback.format_exc())
                print(f"Error processing packet {i+1}: {e}")
                # Append NaN for both predictions if error occurs
                adversarial_packets.append((pkt, 1))
                orig_re.append(np.nan)
                adv_re.append(np.nan)
                continue

        # Total original time
        original_time = original_end_time - original_start_time

        # Get the directory path for the output file
        folder_path = os.path.dirname(self.adv_pcap_path)
        os.makedirs(folder_path, exist_ok=True)
        packets_to_write = [p[0] for p in adversarial_packets]
        wrpcap(self.adv_pcap_path, packets_to_write)

        print(f"Epsilon value for the attack:\t\t\t\t{self.epsilon}")
        print(f"Time taken for the attack:\t\t\t\t{time.time() - start}")
        
        # Print comparison statistics
        print(f"Original predictions computed: {len(orig_re)}")
        print(f"Adversarial predictions computed: {len(adv_re)}")
        
        # Return both original and adversarial results
        return adversarial_packets, adv_re, original_time, orig_re
