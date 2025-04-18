import numpy as np

def s_box_4bit(input_bits):
    mapping = {
        '0000': '1000',
        '0001': '0100',
        '0010': '0010',
        '0011': '1100',
        '0100': '0001',
        '0101': '0101',
        '0110': '1011',
        '0111': '1111',
        '1000': '1001',
        '1001': '0011',
        '1010': '0110',
        '1011': '1010',
        '1100': '0111',
        '1101': '1101',
        '1110': '0000',
        '1111': '1110'
    }
    return mapping[input_bits]

def inverse_s_box_4bit(input_bits):
    mapping = {
        '1000': '0000',
        '0100': '0001',
        '0010': '0010',
        '1100': '0011',
        '0001': '0100',
        '0101': '0101',
        '1011': '0110',
        '1111': '0111',
        '1001': '1000',
        '0011': '1001',
        '0110': '1010',
        '1010': '1011',
        '0111': '1100',
        '1101': '1101',
        '0000': '1110',
        '1110': '1111'
    }
    return mapping[input_bits]

def shift_rows(data):
    shifted = list(data)
    shifted[4:8], shifted[12:16] = data[12:16], data[4:8]
    return shifted

def inverse_shift_rows(data):
    shifted = list(data)
    shifted[12:16], shifted[4:8] = data[4:8], data[12:16]
    return shifted

def mix_columns(data):
    mixed = list(data)
    
    mixed[0] = data[0] ^ data[6]
    mixed[3] = data[3] ^ data[5]
    mixed[2] = data[2] ^ data[4]
    mixed[7] = data[7] ^ data[1]
    mixed[4] = data[4] ^ data[7]
    mixed[5] = data[5] ^ data[2]
    mixed[0] = data[0] ^ data[3]
    mixed[1] = data[1] ^ data[6]
    
    mixed[0], mixed[2] = mixed[2], mixed[0]
    mixed[1], mixed[4] = mixed[4], mixed[1]
    mixed[2], mixed[5] = mixed[5], mixed[2]
    mixed[4], mixed[6] = mixed[6], mixed[4]
    
    mixed[8] = data[8] ^ data[14]
    mixed[11] = data[11] ^ data[13]
    mixed[10] = data[10] ^ data[12]
    mixed[15] = data[15] ^ data[9]
    mixed[12] = data[12] ^ data[15]
    mixed[13] = data[13] ^ data[10]
    mixed[8] = data[8] ^ data[11]
    mixed[9] = data[9] ^ data[14]
    
    mixed[8], mixed[10] = mixed[10], mixed[8]
    mixed[9], mixed[12] = mixed[12], mixed[9]
    mixed[10], mixed[13] = mixed[13], mixed[10]
    mixed[12], mixed[14] = mixed[14], mixed[12]
    
    return mixed

def inverse_mix_columns(data):
    mixed = list(data)
    
    mixed[4], mixed[6] = mixed[6], mixed[4]
    mixed[2], mixed[5] = mixed[5], mixed[2]
    mixed[1], mixed[4] = mixed[4], mixed[1]
    mixed[0], mixed[2] = mixed[2], mixed[0]
    
    mixed[1] = data[1] ^ data[6]
    mixed[0] = data[0] ^ data[3]
    mixed[5] = data[5] ^ data[2]
    mixed[4] = data[4] ^ data[7]
    mixed[7] = data[7] ^ data[1]
    mixed[2] = data[2] ^ data[4]
    mixed[3] = data[3] ^ data[5]
    mixed[6] = data[6] ^ data[0]
    
    mixed[12], mixed[14] = mixed[14], mixed[12]
    mixed[10], mixed[13] = mixed[13], mixed[10]
    mixed[9], mixed[12] = mixed[12], mixed[9]
    mixed[8], mixed[10] = mixed[10], mixed[8]
    
    mixed[9] = data[9] ^ data[14]
    mixed[8] = data[8] ^ data[11]
    mixed[13] = data[13] ^ data[10]
    mixed[12] = data[12] ^ data[15]
    mixed[15] = data[15] ^ data[9]
    mixed[10] = data[10] ^ data[12]
    mixed[11] = data[11] ^ data[13]
    mixed[14] = data[14] ^ data[8]
    
    return mixed

def add_round_key(data, key):
    return [data[i] ^ key[i] for i in range(len(data))]

def key_expansion(key, constant):
    W0 = key[:8]
    W1 = key[8:16]
    
    constant_bits = [int(bit) for bit in constant]
    
    W0_new = [W0[i] ^ constant_bits[i] for i in range(8)]
    
    W1_new = W1.copy()
    W1_new[0:4], W1_new[4:8] = W1_new[4:8], W1_new[0:4]
    
    W1_new_first = ''.join(str(bit) for bit in W1_new[0:4])
    W1_new_second = ''.join(str(bit) for bit in W1_new[4:8])
    
    W1_new_first = s_box_4bit(W1_new_first)
    W1_new_second = s_box_4bit(W1_new_second)
    
    W1_new[0:4] = [int(bit) for bit in W1_new_first]
    W1_new[4:8] = [int(bit) for bit in W1_new_second]
    
    W0_new = [W0_new[i] ^ W1_new[i] for i in range(8)]
    
    W1_new_first = ''.join(str(bit) for bit in W1_new[0:4])
    W1_new_second = ''.join(str(bit) for bit in W1_new[4:8])
    
    W1_new_first = inverse_s_box_4bit(W1_new_first)
    W1_new_second = inverse_s_box_4bit(W1_new_second)
    
    W1_new[0:4] = [int(bit) for bit in W1_new_first]
    W1_new[4:8] = [int(bit) for bit in W1_new_second]
    
    W1_new[0:4], W1_new[4:8] = W1_new[4:8], W1_new[0:4]
    
    W1_new = [W0_new[i] ^ W1_new[i] for i in range(8)]
    
    return W0_new + W1_new

def encrypt_spn(plaintext, key):
    plaintext_bits = [int(bit) for bit in plaintext]
    key_bits = [int(bit) for bit in key]
    
    state = add_round_key(plaintext_bits, key_bits)
    
    for i in range(0, 16, 4):
        block = ''.join(str(bit) for bit in state[i:i+4])
        substituted = s_box_4bit(block)
        state[i:i+4] = [int(bit) for bit in substituted]
    
    state = shift_rows(state)
    
    state = mix_columns(state)
    
    key_bits = key_expansion(key_bits, "10000000")
    
    state = add_round_key(state, key_bits)
    
    for i in range(0, 16, 4):
        block = ''.join(str(bit) for bit in state[i:i+4])
        substituted = s_box_4bit(block)
        state[i:i+4] = [int(bit) for bit in substituted]
    
    state = shift_rows(state)
    
    key_bits = key_expansion(key_bits, "00110000")
    
    state = add_round_key(state, key_bits)
    
    ciphertext = ''.join(str(bit) for bit in state)
    return ciphertext

def decrypt_spn(ciphertext, key):
    ciphertext_bits = [int(bit) for bit in ciphertext]
    key_bits = [int(bit) for bit in key]
    
    key_bits_round1 = key_expansion(key_bits, "10000000")
    key_bits_round2 = key_expansion(key_bits_round1, "00110000")
    
    state = add_round_key(ciphertext_bits, key_bits_round2)
    
    state = inverse_shift_rows(state)
    
    for i in range(0, 16, 4):
        block = ''.join(str(bit) for bit in state[i:i+4])
        substituted = inverse_s_box_4bit(block)
        state[i:i+4] = [int(bit) for bit in substituted]
    
    state = add_round_key(state, key_bits_round1)
    
    state = inverse_mix_columns(state)
    
    state = inverse_shift_rows(state)
    
    for i in range(0, 16, 4):
        block = ''.join(str(bit) for bit in state[i:i+4])
        substituted = inverse_s_box_4bit(block)
        state[i:i+4] = [int(bit) for bit in substituted]
    
    state = add_round_key(state, key_bits)
    
    plaintext = ''.join(str(bit) for bit in state)
    return plaintext

if __name__ == "__main__":
    plaintext = "1101101100100101"
    key = "0011001000110100"
    
    ciphertext = encrypt_spn(plaintext, key)
    print(ciphertext)