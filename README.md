## Surrogate Model

### Environment Details
- OS: Ubuntu 20.04.5 LTS
- Python version: 3.10.14

### Environment Setup
1. Go to the project root directory.
2. Create a virtual environment:
   ```bash
   conda create -n ghosturb python=3.10.14
   conda activate ghosturb
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Ghosturb: (Execute inside the ./code folder)

1. **Train the Model (Surrogate)**
    ```bash
    python surrogate.py --pcap ../data/uq-iot/pcaps/benign/weekday_100k.pcap --mode train --device cuda --dataset uq-iot
    ```

2. **Evaluation (Surrogate)**
   ```bash
   python surrogate.py --pcap ../data/uq-iot/pcaps/malicious/SYN_Flooding.pcap --mode eval --model_path ../artifacts/uq-iot/models/autoencoder.pth --device cuda --dataset uq-iot
   ```
   
3. **Generate Adversarial Examples**
   ```bash
   python main.py --attack ghosturb --threshold 0.130 --pcap-path ../data/uq-iot/pcaps/malicious/SYN_Flooding.pcap --device cpu --epsilon 0.00005 --dataset uq-iot --spoof-ip
   ```
