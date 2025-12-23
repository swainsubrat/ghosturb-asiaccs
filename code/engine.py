"""
This containst core utility functions for algorithims
"""
from scapy.all import *
import numpy as np
import random
import math
import copy
from constant import ip_pools

def get_perturbation(queue, threshold, max_perturbation, method='heuristics'):
    """
    Dispatcher function to call the appropriate projection method.
    
    Args:
        queue: Deque containing history of anomaly scores and timestamps
        threshold: Detection threshold
        max_perturbation: Maximum allowed time perturbation (epsilon)
        method: Projection method to use. Options:
            - 'distance': Simple distance-based (reactive)
            - 'linear': Linear extrapolation (1st order Taylor)
            - 'heuristics' or 'quadratic': Quadratic projection (2nd order Taylor)
            - 'cubic': Cubic projection (3rd order Taylor)
            - 'kalman': Kalman filter based
            - 'state_space': State space model based (default)
            - 'arima': ARIMA statistical forecasting
    
    Returns:
        float: Calculated time perturbation value
    """
    method = method.lower()
    
    # Map method names to functions
    method_map = {
        'heuristics': get_perturbation_heuristics,
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown projection method: {method}. Valid options: {list(method_map.keys())}")
    
    # Call the appropriate function
    return method_map[method](queue, threshold, max_perturbation)


def calculate_perturbation(time_required: float,
                          max_perturbation: float,
                          min_perturbation: float=1e-5,
                          k=10000) -> float:

    time_perturbation = min_perturbation + math.tanh(1/(k * time_required)) * (max_perturbation - min_perturbation)

    return time_perturbation

def scale_value(value, min_range, max_range):
    return min_range + (value * (max_range - min_range))

def solve_quadratic(distance, velocity, acceleration):
    """
    Helper funciton to solve quadratic equation
    """
    a = 0.5 * acceleration
    b = velocity
    c = -distance

    # Use scimath.sqrt to allow for complex results
    discriminant = b**2 - 4*a*c
    sqrt_disc = np.lib.scimath.sqrt(discriminant)
    t1 = (-b + sqrt_disc) / (2*a)
    t2 = (-b - sqrt_disc) / (2*a)

    return discriminant, t1, t2

def get_perturbation_heuristics(queue, threshold, max_perturbation=0.001):
    """
    Assumptions:
    Distance is p% of the threshold
    If prev_timegap is 0, 1e-5 is assigned
    If acceleration is 0, take velocity
    """
    min_time_perturbation = 1e-5
    # min_time_perturbation = 0

    time_required = None
    time_perturbation = None

    curr_pos = queue[-1]["anomaly_score"]
    prev_pos = queue[-2]["anomaly_score"]
    past_pos = queue[-3]["anomaly_score"]

    # distance caluculation
    distance = (0.75 * threshold) - curr_pos

    # case 1: max perturbation
    if distance <= 0: # already above the threshold line, return max perturbation
        # print(f"Already above threshold!!!!: {curr_pos} >= {0.9 * threshold}")
        return max_perturbation

    curr_timestamp = float(queue[-1]["timestamp"])
    prev_timestamp = float(queue[-2]["timestamp"])
    past_timestamp = float(queue[-3]["timestamp"])
    prev_distance = curr_pos - prev_pos
    prev_timegap  = curr_timestamp - prev_timestamp
    
    # added this line because it was showing division by float 0 at prev_timegap
    # velocity calculation
    if prev_timegap == 0:
        prev_timegap = 1e-5
    velocity = prev_distance / prev_timegap

    if velocity == 0:
        print("Zero velocity found!!!!")
        return min_time_perturbation
    
    past_distance = prev_pos - past_pos
    past_timegap = prev_timestamp - past_timestamp

    # add this line becuase it was showing division by float 0 at past_timegap
    if past_timegap == 0:
        past_timegap = 1e-5
    past_velocity = past_distance / past_timegap

    # # acceleration calculation # TODO: check this never happens
    if (past_timegap + prev_timegap) == 0:
        past_timegap = prev_timegap = 1e-5
    acceleration = (velocity - past_velocity) / (past_timegap + prev_timegap)

    # if acceleration is zero, consider constant velocity
    if acceleration == 0:
        print("Considering constant speed!!!!")
        if velocity > 0:
            time_required = distance / velocity

    else:
        discriminant, time_required_1, time_required_2 = solve_quadratic(
            distance, velocity, acceleration)

        # first evaluate the complex case
        if discriminant < 0:
            # confirm maxima
            if 0.5 * acceleration > 0:
                print("[Alert!] This should not happen, as giving complex soultion \
                    below threshold, the parabola should be downward facing")
            
            # find maxima
            time_vertex = -velocity / acceleration
            
            if time_vertex < curr_timestamp:
                time_perturbation = min_time_perturbation
            
            else:
                time_required = time_vertex
        
        else:
            # case 2: calculate t_req
            # case 2.1: if both positive, then take minimum
            if time_required_1 > 0 and time_required_2 > 0:
                time_required = min(time_required_1, time_required_2)

            # case 2.2: if atleast one positive, take the positive one
            # elif (time_required_1 > 0 and time_required_2 < 0) or (time_required_1 < 0 and time_required_2 > 0):
            elif time_required_1 > 0:
                time_required = time_required_1

                # confirmimg minima
                if 0.5 * acceleration < 0:
                    print("[Alert!] This should not happen, as there is one +ve and \
                        one -ve, the parabola should be upward facing")

            # case 3: if both negetive, then perturb minimum or none (discuss and update later)
            elif time_required_1 < 0 and time_required_2 < 0:
                time_perturbation = min_time_perturbation

            else:
                print("This case shouldn't happen!!!!! Inspect!!!!")
                import pdb; pdb.set_trace()
    
    if time_required is not None:

        # # append the time required to a file that is already opened
        # with open("./at.csv", "a") as f:
        #     f.write(f"{time_required}\n")

        if time_perturbation is not None:
            print("This should not happen as the time_perturbation can't be calculated!!!!")
        
        # time_perturbation_multiplier = (2 * sigmoid((1/time_required)) - 1)
        # time_perturbation = scale_value(time_perturbation_multiplier, 0, max_perturbation)

        time_perturbation = calculate_perturbation(time_required, max_perturbation, min_time_perturbation)

    return time_perturbation

def change_sport(packet):

    # Generate a random port number in the range 1024 to 65535 (dynamic/private ports)
    new_port = random.randint(1024, 65535)
    # new_port=80
    # Check if the packet has a TCP or UDP layer
    if packet.haslayer(TCP):
        packet[TCP].sport = new_port
    elif packet.haslayer(UDP):
        packet[UDP].sport = new_port
    else:
        return packet

    # Recalculate the checksums after modification
    if packet.haslayer(IP):
        del packet[IP].chksum  # Delete IP checksum to force recalculation
    if packet.haslayer(TCP):
        del packet[TCP].chksum  # Delete TCP checksum
    elif packet.haslayer(UDP):
        del packet[UDP].chksum  # Delete UDP checksum

    return packet

def change_dport(packet):

    # Generate a random port number in the range 1024 to 65535 (dynamic/private ports)
    # new_port = random.randint(1024, 65535)
    new_port=53127
    # Check if the packet has a TCP or UDP layer
    if packet.haslayer(TCP):
        packet[TCP].dport = new_port
    elif packet.haslayer(UDP):
        packet[UDP].dport = new_port
    else:
        print("Packet does not have a TCP or UDP layer. Port change is not applicable.")
        return packet

    # Recalculate the checksums after modification
    del packet[IP].chksum  # Delete IP checksum to force recalculation
    if packet.haslayer(TCP):
        del packet[TCP].chksum  # Delete TCP checksum
    elif packet.haslayer(UDP):
        del packet[UDP].chksum  # Delete UDP checksum

    return packet

def change_ip(packet):
    if packet.haslayer(IP):
            spoofed_ip = random.choice(ip_pools)
            packet[IP].src = spoofed_ip
            del packet[IP].len
            del packet[IP].chksum
    return packet

def add_raw_packet(packet, payload_size):

    r = 'z'
    size=payload_size
    if len(packet) != 0:
            if packet.haslayer(TCP):
                packet[TCP].remove_payload()
                payload_size = len(packet) + size
                packet[TCP].add_payload(Raw(load="b" * payload_size))
                del packet[TCP].chksum  # Remove TCP checksum

            if packet.haslayer(UDP):
                packet[UDP].remove_payload()
                payload_size = len(packet) + size
                packet[UDP].add_payload(Raw(load="c" * payload_size))

            if packet.haslayer(IP):
                del packet[IP].len  # Remove IP length
                del packet[IP].chksum  # Remove IP checksum
                del packet.len


    return packet

def size_change_perturbation():
    
    return random.randint(-1200,1800)

def get_insertion_packets(pkt, timestamps, size):

    insertion_packets = []
    for ts in timestamps:
        packet = copy.deepcopy(pkt)
        packet.time = ts

        packet = add_raw_packet(packet, size)

        insertion_packets.append(packet)
    
    return insertion_packets
