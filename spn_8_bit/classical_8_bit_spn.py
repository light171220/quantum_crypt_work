def binary_to_int(binary_str):
    return int(binary_str, 2)

def int_to_binary(n, bits=8):
    return format(n, f'0{bits}b')

def encrypt_yoyo(plaintext, key):
    sbox = [0xc, 5, 6, 0xb, 9, 0, 0xa, 0xd, 3, 0xe, 0xf, 8, 4, 7, 1, 2]
    
    pt = binary_to_int(plaintext)
    k1 = binary_to_int(key)
    
    key1 = (k1 >> 4) & 0xF
    key2 = k1 & 0xF
    
    key1 = sbox[key1]
    key2 = sbox[key2]
    
    temp0 = (key1 >> 0) & 1
    temp1 = (key1 >> 1) & 1
    temp2 = (key1 >> 2) & 1
    temp3 = (key1 >> 3) & 1
    
    key1 = (temp3 << 0) | (temp2 << 1) | (temp0 << 2) | (temp1 << 3)
    
    temp0 = (key2 >> 0) & 1
    temp1 = (key2 >> 1) & 1
    temp2 = (key2 >> 2) & 1
    temp3 = (key2 >> 3) & 1
    
    key2 = (temp3 << 0) | (temp2 << 1) | (temp0 << 2) | (temp1 << 3)
    
    k2 = (key1 << 4) | key2
    
    pt ^= k1
    
    pt_high = (pt >> 4) & 0xF
    pt_low = pt & 0xF
    pt = (sbox[pt_high] << 4) | sbox[pt_low]
    
    temp0 = (pt >> 0) & 1
    temp1 = (pt >> 1) & 1
    temp2 = (pt >> 2) & 1
    temp3 = (pt >> 3) & 1
    temp4 = (pt >> 4) & 1
    temp5 = (pt >> 5) & 1
    temp6 = (pt >> 6) & 1
    temp7 = (pt >> 7) & 1
    
    pt = (temp7 << 0) | (temp2 << 1) | (temp4 << 2) | (temp6 << 3) | \
         (temp1 << 4) | (temp3 << 5) | (temp5 << 6) | (temp0 << 7)
    
    pt ^= k2
    
    pt_high = (pt >> 4) & 0xF
    pt_low = pt & 0xF
    pt = (sbox[pt_high] << 4) | sbox[pt_low]
    
    temp0 = (pt >> 0) & 1
    temp1 = (pt >> 1) & 1
    temp2 = (pt >> 2) & 1
    temp3 = (pt >> 3) & 1
    temp4 = (pt >> 4) & 1
    temp5 = (pt >> 5) & 1
    temp6 = (pt >> 6) & 1
    temp7 = (pt >> 7) & 1
    
    pt = (temp7 << 0) | (temp2 << 1) | (temp4 << 2) | (temp6 << 3) | \
         (temp1 << 4) | (temp3 << 5) | (temp5 << 6) | (temp0 << 7)
    
    return int_to_binary(pt)

def main():
    plaintext = input("Enter 8-bit plaintext (e.g., 11011011): ")
    key = input("Enter 8-bit key (e.g., 00010010): ")
    
    if len(plaintext) != 8 or not all(bit in '01' for bit in plaintext):
        print("Error: Plaintext must be exactly 8 bits (0s and 1s)")
        return
    
    if len(key) != 8 or not all(bit in '01' for bit in key):
        print("Error: Key must be exactly 8 bits (0s and 1s)")
        return
    
    ciphertext = encrypt_yoyo(plaintext, key)
    print(f"\nPlaintext:  {plaintext}")
    print(f"Key:        {key}")
    print(f"Ciphertext: {ciphertext}")

if __name__ == "__main__":
    main()