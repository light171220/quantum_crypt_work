def permute(bits, pattern):
    return [bits[i-1] for i in pattern]

def left_shift(bits, n):
    return bits[n:] + bits[:n]

def generate_subkeys(key):
    p10 = [3, 5, 2, 7, 4, 10, 1, 9, 8, 6]
    p8 = [6, 3, 7, 4, 8, 5, 10, 9]
    key_permuted = permute(key, p10)
    left, right = key_permuted[:5], key_permuted[5:]
    subkeys = []
    for shift in [1, 2]:
        left, right = left_shift(left, shift), left_shift(right, shift)
        subkeys.append(permute(left + right, p8))
    return subkeys

def f_function(right, subkey):
    e_p = [4, 1, 2, 3, 2, 3, 4, 1]
    p4 = [2, 4, 3, 1]
    right_expanded = permute(right, e_p)
    xor_result = [r ^ k for r, k in zip(right_expanded, subkey)]
    s0 = [
        [1, 0, 3, 2],
        [3, 2, 1, 0],
        [0, 2, 1, 3],
        [3, 1, 2, 0]
    ]
    s1 = [
        [0, 2, 1, 3],
        [2, 0, 1, 3],
        [3, 2, 1, 0],
        [2, 1, 0, 3]
    ]

    def s_box(bits, box):
        row = bits[0] * 2 + bits[3]
        col = bits[1] * 2 + bits[2]
        return [int(x) for x in f"{box[row][col]:02b}"]
    
    left, right = xor_result[:4], xor_result[4:]
    sbox_result = s_box(left, s0) + s_box(right, s1)
    return permute(sbox_result, p4)

def sdes_encrypt(plaintext, key):
    ip = [2, 6, 3, 1, 4, 8, 5, 7]
    ip_inv = [4, 1, 3, 5, 7, 2, 8, 6]
    subkeys = generate_subkeys(key)
    bits = permute(plaintext, ip)
    
    def fk(bits, subkey):
        left, right = bits[:4], bits[4:]
        return [l ^ r for l, r in zip(left, f_function(right, subkey))] + right
    
    bits = fk(bits, subkeys[0])
    bits = bits[4:] + bits[:4]  # SW function (swap)
    bits = fk(bits, subkeys[1])
    return permute(bits, ip_inv)

def sdes_decrypt(ciphertext, key):
    ip = [2, 6, 3, 1, 4, 8, 5, 7]
    ip_inv = [4, 1, 3, 5, 7, 2, 8, 6]
    subkeys = generate_subkeys(key)
    bits = permute(ciphertext, ip)
    
    def fk(bits, subkey):
        left, right = bits[:4], bits[4:]
        return [l ^ r for l, r in zip(left, f_function(right, subkey))] + right
    
    bits = fk(bits, subkeys[1])
    bits = bits[4:] + bits[:4]
    bits = fk(bits, subkeys[0])
    return permute(bits, ip_inv)

def str_to_bits(s):
    return [int(bit) for bit in s]

def bits_to_str(bits):
    return ''.join(map(str, bits))

def validate_binary(s, length):
    if not all(bit in '01' for bit in s):
        return False
    if len(s) != length:
        return False
    return True

def main():
    print("=" * 50)
    print("SIMPLIFIED DES ENCRYPTION AND DECRYPTION")
    print("=" * 50)
    
    while True:
        plaintext_str = input("Enter 8-bit plaintext (binary): ")
        if validate_binary(plaintext_str, 8):
            plaintext = str_to_bits(plaintext_str)
            break
        print("Invalid input. Please enter exactly 8 binary digits (0 or 1).")
    
    while True:
        key_str = input("Enter 10-bit key (binary): ")
        if validate_binary(key_str, 10):
            key = str_to_bits(key_str)
            break
        print("Invalid input. Please enter exactly 10 binary digits (0 or 1).")
    
    ciphertext = sdes_encrypt(plaintext, key)
    decrypted = sdes_decrypt(ciphertext, key)
    
    print("\nResults:")
    print(f"Plaintext:  {bits_to_str(plaintext)}")
    print(f"Key:        {bits_to_str(key)}")
    print(f"Ciphertext: {bits_to_str(ciphertext)}")
    print(f"Decrypted:  {bits_to_str(decrypted)}")

if __name__ == "__main__":
    main()