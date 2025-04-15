def binary_to_int(binary_str):
    return int(binary_str, 2)

def int_to_binary(n, bits=8):
    return format(n, f'0{bits}b')

def xor_bits(bits1, bits2):
    result = ""
    for b1, b2 in zip(bits1, bits2):
        result += "1" if b1 != b2 else "0"
    return result

def sbox_lookup(input_bits):
    sbox = {
        "0000": "1100", # 0 -> C
        "0001": "0101", # 1 -> 5
        "0010": "0110", # 2 -> 6
        "0011": "1011", # 3 -> B
        "0100": "1001", # 4 -> 9
        "0101": "0000", # 5 -> 0
        "0110": "1010", # 6 -> A
        "0111": "1101", # 7 -> D
        "1000": "0011", # 8 -> 3
        "1001": "1110", # 9 -> E
        "1010": "1111", # A -> F
        "1011": "1000", # B -> 8
        "1100": "0100", # C -> 4
        "1101": "0111", # D -> 7
        "1110": "0001", # E -> 1
        "1111": "0010"  # F -> 2
    }
    return sbox[input_bits]

def inverse_sbox_lookup(output_bits):
    inverse_sbox = {
        "1100": "0000", # C -> 0
        "0101": "0001", # 5 -> 1
        "0110": "0010", # 6 -> 2
        "1011": "0011", # B -> 3
        "1001": "0100", # 9 -> 4
        "0000": "0101", # 0 -> 5
        "1010": "0110", # A -> 6
        "1101": "0111", # D -> 7
        "0011": "1000", # 3 -> 8
        "1110": "1001", # E -> 9
        "1111": "1010", # F -> A
        "1000": "1011", # 8 -> B
        "0100": "1100", # 4 -> C
        "0111": "1101", # 7 -> D
        "0001": "1110", # 1 -> E
        "0010": "1111"  # 2 -> F
    }
    return inverse_sbox[output_bits]

def apply_sbox(bits):
    high_nibble = bits[:4]
    low_nibble = bits[4:]
    return sbox_lookup(high_nibble) + sbox_lookup(low_nibble)

def apply_inverse_sbox(bits):
    high_nibble = bits[:4]
    low_nibble = bits[4:]
    return inverse_sbox_lookup(high_nibble) + inverse_sbox_lookup(low_nibble)

def permutation_p(bits):
    p_map = [7, 4, 1, 5, 2, 6, 3, 0]
    result = ["0"] * 8
    for i, bit in enumerate(bits):
        result[p_map[i]] = bit
    return "".join(result)

def inverse_permutation_p(bits):
    p_map = [7, 2, 4, 6, 1, 3, 5, 0]
    result = ["0"] * 8
    for i, bit in enumerate(bits):
        result[p_map[i]] = bit
    return "".join(result)

def key_permutation_pk(bits):
    pk_map = [1, 0, 2, 3]
    result = ["0"] * 4
    for i, bit in enumerate(bits):
        result[pk_map[i]] = bit
    return "".join(result)

def inverse_key_permutation_pk(bits):
    inv_pk_map = [1, 0, 2, 3]
    result = ["0"] * 4
    for i, bit in enumerate(bits):
        result[inv_pk_map[i]] = bit
    return "".join(result)

def key_generation(key):
    high_nibble = key[:4]
    low_nibble = key[4:]
    
    high_nibble_sbox = sbox_lookup(high_nibble)
    low_nibble_sbox = sbox_lookup(low_nibble)
    
    high_nibble_permuted = key_permutation_pk(high_nibble_sbox)
    low_nibble_permuted = key_permutation_pk(low_nibble_sbox)
    
    return high_nibble_permuted + low_nibble_permuted

def inverse_key_generation(key2):
    high_nibble = key2[:4]
    low_nibble = key2[4:]
    
    high_nibble_inv_permuted = inverse_key_permutation_pk(high_nibble)
    low_nibble_inv_permuted = inverse_key_permutation_pk(low_nibble)
    
    high_nibble_inv_sbox = inverse_sbox_lookup(high_nibble_inv_permuted)
    low_nibble_inv_sbox = inverse_sbox_lookup(low_nibble_inv_permuted)
    
    return high_nibble_inv_sbox + low_nibble_inv_sbox

def encrypt_round(plaintext, key):
    xored = xor_bits(plaintext, key)
    sboxed = apply_sbox(xored)
    permuted = permutation_p(sboxed)
    return permuted

def encrypt_yoyo(plaintext, key):
    round_key1 = key
    round_key2 = key_generation(key)
    
    after_round1 = encrypt_round(plaintext, round_key1)
    ciphertext = encrypt_round(after_round1, round_key2)
    
    return ciphertext

def decrypt_round(ciphertext, key):
    inv_permuted = inverse_permutation_p(ciphertext)
    inv_sboxed = apply_inverse_sbox(inv_permuted)
    inv_xored = xor_bits(inv_sboxed, key)
    return inv_xored

def decrypt_yoyo(ciphertext, key):
    round_key1 = key
    round_key2 = key_generation(key)
    
    after_round1 = decrypt_round(ciphertext, round_key2)
    plaintext = decrypt_round(after_round1, round_key1)
    
    return plaintext

def find_key(plaintext, ciphertext):
    all_keys = []
    for i in range(256):
        key = int_to_binary(i, 8)
        if encrypt_yoyo(plaintext, key) == ciphertext:
            all_keys.append(key)
    return all_keys

def test_yoyo_cipher():
    print("Testing Yo-yo Block Cipher")
    print("--------------------------")
    
    test_cases = [
        ("11011011", "00100010"),
        ("11001100", "01011010"),
        ("10101010", "11110000"),
        ("11111111", "00000000")
    ]
    
    for plaintext, ciphertext in test_cases:
        print(f"\nTest case: Plaintext={plaintext}, Ciphertext={ciphertext}")
        
        keys = find_key(plaintext, ciphertext)
        if keys:
            print(f"Found {len(keys)} key(s):")
            for key in keys:
                print(f"Key: {key}")
                encrypted = encrypt_yoyo(plaintext, key)
                decrypted = decrypt_yoyo(ciphertext, key)
                print(f"Verification - Encrypted: {encrypted}, Decrypted: {decrypted}")
                print(f"Match ciphertext: {encrypted == ciphertext}, Match plaintext: {decrypted == plaintext}")
        else:
            print("No keys found for this plaintext-ciphertext pair")

def test_encrypt_decrypt_cycle(key):
    print(f"\nTesting encrypt/decrypt cycle with key: {key}")
    print("-------------------------------------------")
    
    import random
    
    for _ in range(5):
        plaintext = ''.join(random.choice('01') for _ in range(8))
        ciphertext = encrypt_yoyo(plaintext, key)
        decrypted = decrypt_yoyo(ciphertext, key)
        
        print(f"Plaintext: {plaintext}")
        print(f"Ciphertext: {ciphertext}")
        print(f"Decrypted: {decrypted}")
        print(f"Cycle works: {plaintext == decrypted}\n")

def generate_test_pairs(key, num_pairs=5):
    print(f"\nGenerating test pairs using key: {key}")
    print("----------------------------------------")
    
    import random
    pairs = []
    
    for _ in range(num_pairs):
        plaintext = ''.join(random.choice('01') for _ in range(8))
        ciphertext = encrypt_yoyo(plaintext, key)
        pairs.append((plaintext, ciphertext))
        print(f"Plaintext: {plaintext} -> Ciphertext: {ciphertext}")
    
    return pairs

if __name__ == "__main__":
    test_yoyo_cipher()
    
    # Test the encrypt/decrypt cycle
    test_key = "00010010"  # 18 in decimal
    test_encrypt_decrypt_cycle(test_key)
    
    # Generate some test pairs with a known key
    generate_test_pairs(test_key, 10)