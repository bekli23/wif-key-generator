# WIF Key Generator with Public Key Verification

This project is a Python-based tool for generating Bitcoin Wallet Import Format (WIF) keys and verifying their corresponding public keys. The tool uses CUDA for parallel key generation and includes various modes to customize the key generation process.

## Features

- **WIF Key Generation**: Generate WIF keys with customizable patterns.
- **Public Key Verification**: Check if the generated WIF corresponds to a specific target public key.
- **Multiple Modes**: Six different modes for generating WIF keys.
- **Real-Time Output**: Displays the generated WIF, private key in HEX format, and the corresponding public key in real-time.
- **BSGS Attack Integration**: Uses the Baby-Step Giant-Step algorithm for faster key verification.
- **Error Handling**: Automatically corrects and verifies WIF keys with invalid checksums.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/bekli23/wif-key-generator.git
   cd wif-key-generator
Install the required Python packages:

Make sure you have Python 3.8 or higher installed.

pip install pycuda colorama ecdsa
You may need to install additional CUDA dependencies if not already installed.

Run the script:

python cov.py
Usage
Enter the k value based on your available RAM:

RAM	k Value
2 GB	128
4 GB	256
8 GB	512
16 GB	1024
32 GB	2048

Select the generation mode:

1: Random Mode
2: Normal Mode
3: Specific Mode
4: Incremental Mode
5: Reversed Mode
6: Alternating Mode
Example:

plaintext
Copy code
Select generation mode:
1. Random
2. Normal
3. Specific
4. Incremental
5. Reversed
6. Alternating
Mode: 1
Monitor the output to see the generated WIF, private key in HEX format, and the public key:

Generated WIF: L22899D3vz**********g***JN9pUefQAPDGXJysxp1UvBg2t3
Private Key (HEX): 1b2c3d4e5f67890a...
Public Key: 0252b0c8673488f07dd67a9c5555e78c5e9aca537661438c523bd26ec7ebc1d3c6
Generation Speed: 1.23 MH/s
Stop the script by pressing CTRL+C.
