import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import pycuda.compiler as compiler
import time
import sys
import threading
import queue
from colorama import init, Fore, Style
import hashlib
import base58
import ecdsa
from math import isqrt

# Initialize colorama
init(autoreset=True)

# Target public key
TARGET_PUBLIC_KEY = "0252b0c8673488f07dd67a9c5555e78c5e9aca537661438c523bd26ec7ebc1d3c6"

# CUDA kernel for generating WIF keys
kernel_code = """
__device__ char base58_chars[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

__global__ void generate_keys(char *wif_key_template, char *result_keys, int asterisks_count, int num_keys, int mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_keys) {
        int key_len = 0;
        while (wif_key_template[key_len] != '\\0') {
            key_len++;
        }

        int base58_chars_len = 58;

        for (int i = 0; i < key_len; i++) {
            if (wif_key_template[i] == '*') {
                int char_idx;
                if (mode == 1) {  // Random mode
                    char_idx = (idx + i) % base58_chars_len;
                } else if (mode == 2) {  // Normal mode
                    char_idx = (idx + i) % base58_chars_len;
                } else if (mode == 3) {  // Specific mode
                    char_idx = (idx + 1 + i) % base58_chars_len;
                } else if (mode == 4) {  // Incremental mode
                    char_idx = (idx + i + 1) % base58_chars_len;
                } else if (mode == 5) {  // Reversed mode
                    char_idx = (base58_chars_len - 1 - (idx + i) % base58_chars_len);
                } else if (mode == 6) {  // Alternating mode
                    char_idx = (idx % 2 == 0) ? (i % base58_chars_len) : (base58_chars_len - 1 - i);
                } else {
                    char_idx = (idx + i) % base58_chars_len;  // Default to random if mode is out of bounds
                }
                result_keys[idx * key_len + i] = base58_chars[char_idx];
            } else {
                result_keys[idx * key_len + i] = wif_key_template[i];
            }
        }
    }
}
"""

def private_key_to_wif(private_key_hex, compressed=True):
    private_key_bytes = bytes.fromhex(private_key_hex)
    extended_key = b'\x80' + private_key_bytes

    if compressed:
        extended_key += b'\x01'

    first_sha256 = hashlib.sha256(extended_key).digest()
    second_sha256 = hashlib.sha256(first_sha256).digest()
    checksum = second_sha256[:4]

    final_key = extended_key + checksum
    wif_key = base58.b58encode(final_key).decode('utf-8')
    
    return wif_key

def wif_to_private_key(wif_key):
    try:
        decoded = base58.b58decode(wif_key)
    except ValueError as e:
        raise ValueError(f"Invalid character in WIF: {e}")

    if len(decoded) not in (37, 38):
        raise ValueError("The decoded WIF length is not correct!")

    checksum = decoded[-4:]
    key_data = decoded[:-4]
    first_sha256 = hashlib.sha256(key_data).digest()
    second_sha256 = hashlib.sha256(first_sha256).digest()
    calculated_checksum = second_sha256[:4]

    if checksum != calculated_checksum:
        raise ValueError("Checksum is incorrect, WIF is invalid!")

    private_key_hex = key_data[1:-1].hex() if len(key_data) == 34 else key_data[1:].hex()

    return private_key_hex

def adjust_wif_checksum(wif_key):
    """
    Adjusts the WIF to correct an incorrect checksum.
    """
    decoded = base58.b58decode(wif_key)
    key_data = decoded[:-4]

    # Calculate the correct checksum
    first_sha256 = hashlib.sha256(key_data).digest()
    second_sha256 = hashlib.sha256(first_sha256).digest()
    correct_checksum = second_sha256[:4]

    # Replace the incorrect checksum with the correct one
    corrected_wif_bytes = key_data + correct_checksum
    corrected_wif = base58.b58encode(corrected_wif_bytes).decode('utf-8')
    
    return corrected_wif

def validate_wif_and_adjust(wif_key):
    """
    Validates and adjusts the WIF if the checksum is incorrect.
    """
    try:
        return wif_to_private_key(wif_key)
    except ValueError:
        # If the WIF is invalid, try to adjust it
        corrected_wif = adjust_wif_checksum(wif_key)
        try:
            return wif_to_private_key(corrected_wif)
        except ValueError:
            return None

def calculate_public_key(private_key_hex, compressed=True):
    private_key_bytes = bytes.fromhex(private_key_hex)
    sk = ecdsa.SigningKey.from_string(private_key_bytes, curve=ecdsa.SECP256k1)
    vk = sk.get_verifying_key()
    if compressed:
        public_key = b'\x02' + vk.to_string()[:32] if vk.to_string()[:1] < b'\x80' else b'\x03' + vk.to_string()[:32]
    else:
        public_key = b'\x04' + vk.to_string()
    return public_key.hex()

def format_hashrate(hashes_per_second):
    units = ['H/s', 'kH/s', 'MH/s', 'GH/s', 'TH/s', 'PH/s', 'EH/s', 'ZH/s']
    power = 0
    while hashes_per_second >= 1000 and power < len(units) - 1:
        hashes_per_second /= 1000.0
        power += 1
    return f"{hashes_per_second:.2f} {units[power]}"

def point_to_key(point):
    if point is None or point.x() is None or point.y() is None:
        return None
    return (int(point.x()), int(point.y()))

def bsgs_attack(public_key_hex, max_range, generator, curve, k_value, results_queue, stop_event):
    m = isqrt(max_range) + 1
    baby_steps = {}

    for j in range(m):
        if stop_event.is_set():
            return
        point = j * generator
        if point_to_key(point) is not None:
            baby_steps[point_to_key(point)] = j

    giant_step = m * generator
    pubkey_point = ecdsa.ellipticcurve.Point(curve, int(public_key_hex[:64], 16), int(public_key_hex[64:], 16))
    for i in range(m):
        if stop_event.is_set():
            return
        candidate_point = pubkey_point - i * giant_step
        if point_to_key(candidate_point) in baby_steps:
            results_queue.put(i * m + baby_steps[point_to_key(candidate_point)])
            return

def generate_wif_and_hex(num_keys, wif_key, asterisks_count, generate_keys, max_range, k_value, results_queue, mode, stop_event):
    wif_key_template = np.array([ord(c) for c in wif_key], dtype=np.int8)
    result_keys = np.zeros((num_keys, len(wif_key_template)), dtype=np.int8)

    curve = ecdsa.SECP256k1.curve
    generator = ecdsa.SECP256k1.generator

    combinations_tried = 0
    last_report_time = time.time()

    while not stop_event.is_set():
        start = time.time()
        generate_keys(
            drv.In(wif_key_template), drv.Out(result_keys),
            np.int32(asterisks_count), np.int32(num_keys), np.int32(mode),
            block=(1024, 1, 1), grid=(1, 1)
        )
        end = time.time()

        combinations_tried += num_keys
        current_time = time.time()
        time_elapsed = current_time - last_report_time

        if time_elapsed >= 1.0:
            keys_per_second = combinations_tried / time_elapsed
            formatted_hashrate = format_hashrate(keys_per_second)
            last_report_time = current_time
            combinations_tried = 0

            for i in range(num_keys):
                generated_wif = ''.join([chr(c) for c in result_keys[i] if c != 0])  # Ignore null characters

                # Ensure all asterisks (*) are replaced
                if generated_wif.count('*') != 0:
                    print(f"{Fore.RED}Error: WIF generated with unprocessed '*' characters: {generated_wif}")
                    continue

                try:
                    private_key_hex = validate_wif_and_adjust(generated_wif)
                except ValueError as e:
                    print(f"{Fore.RED}Error: {e}")
                    continue

                if private_key_hex:
                    public_key_hex = calculate_public_key(private_key_hex, compressed=True)

                    # Show real-time information
                    sys.stdout.write(f"{Fore.YELLOW}Generated WIF: {generated_wif}\n")
                    sys.stdout.write(f"{Fore.BLUE}Private Key (Hex): {private_key_hex}\n")
                    sys.stdout.write(f"{Fore.GREEN}Public Key: {public_key_hex}\n")
                    sys.stdout.write(f"{Fore.CYAN}Generation Speed: {formatted_hashrate}\n")
                    sys.stdout.flush()

                    if public_key_hex == TARGET_PUBLIC_KEY:
                        sys.stdout.write(Fore.GREEN + f"Private key found: {private_key_hex}\n")
                        sys.stdout.write(Fore.RED + f"Matching Public Key: {public_key_hex}\n")
                        sys.stdout.flush()
                        stop_event.set()
                        return

                    bsgs_thread = threading.Thread(target=bsgs_attack, args=(public_key_hex, max_range, generator, curve, k_value, results_queue, stop_event))
                    bsgs_thread.start()

def monitor_results(results_queue, stop_event):
    while not stop_event.is_set():
        result = results_queue.get()
        if result:
            sys.stdout.write(Fore.GREEN + f"Private key found by BSGS: {result}\n")
            sys.stdout.flush()
            stop_event.set()
            break

def main():
    wif_key = "L22899D***************g***JN9pUefQAPDGXJysxp1UvBg2t3"
    asterisks_count = wif_key.count('*')
    max_range = 2**64  # Adjust as needed

    # Explanation of k values based on RAM:
    print("Choose the k value based on your available RAM:\n")
    print("2 GB  -> -k 128")
    print("4 GB  -> -k 256")
    print("8 GB  -> -k 512")
    print("16 GB -> -k 1024")
    print("32 GB -> -k 2048")
    print("Adjust k based on your available RAM.\n")

    k_value = int(input("Enter k value (RAM usage): "))
    num_keys = 1024 * k_value  # Adjust the number of generated keys based on k_value

    results_queue = queue.Queue()
    stop_event = threading.Event()

    mod = compiler.SourceModule(kernel_code)
    generate_keys = mod.get_function("generate_keys")

    print("Select generation mode:")
    print("1. Random")
    print("2. Normal")
    print("3. Specific")
    print("4. Incremental")
    print("5. Reversed")
    print("6. Alternating")
    mode = int(input("Mode: "))
    if mode not in [1, 2, 3, 4, 5, 6]:
        mode = 1  # Default to random

    monitor_thread = threading.Thread(target=monitor_results, args=(results_queue, stop_event))
    monitor_thread.start()

    try:
        generate_wif_and_hex(num_keys, wif_key, asterisks_count, generate_keys, max_range, k_value, results_queue, mode, stop_event)
    except KeyboardInterrupt:
        stop_event.set()
        results_queue.put(None)  # Unblock the queue if it's stuck
        sys.stdout.write("Generation stopped by user.\n")
        sys.stdout.flush()

    monitor_thread.join()

if __name__ == "__main__":
    main()
