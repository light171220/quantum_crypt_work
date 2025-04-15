class Cipher32Bit:
    def __init__(self, key):
        if len(key) != 4:
            raise ValueError("Key must be 4 bytes (32 bits)")
        
        self.key = key
        self.rounds = 8
        
        self.s_box = [
            0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
            0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2
        ]
        
        self.inv_s_box = [0] * 16
        for i in range(16):
            self.inv_s_box[self.s_box[i]] = i
            
        self.round_keys = self.generate_round_keys(key)
    
    def generate_round_keys(self, key):
        key_int = int.from_bytes(key, byteorder='big')
        
        round_keys = []
        for i in range(self.rounds):
            round_key = key_int
            round_key = ((round_key << (i + 1)) | (round_key >> (32 - (i + 1)))) & 0xFFFFFFFF
            round_key = round_key ^ (0x01234567 * (i + 1))
            
            round_keys.append(round_key)
            
        return round_keys
    
    def substitute_bytes(self, value, inverse=False):
        result = 0
        s_box_to_use = self.inv_s_box if inverse else self.s_box
        
        for i in range(8):
            nibble = (value >> (4 * i)) & 0xF
            substituted = s_box_to_use[nibble]
            result |= (substituted << (4 * i))
            
        return result
    
    def permute_bits(self, value, inverse=False):
        result = 0
        
        if not inverse:
            for i in range(4):
                byte = (value >> (8 * i)) & 0xFF
                rotated = ((byte << (i + 1)) | (byte >> (8 - (i + 1)))) & 0xFF
                result |= (rotated << (8 * i))
        else:
            for i in range(4):
                byte = (value >> (8 * i)) & 0xFF
                rotated = ((byte >> (i + 1)) | (byte << (8 - (i + 1)))) & 0xFF
                result |= (rotated << (8 * i))
                
        return result
    
    def round_function(self, value, round_key):
        substituted = self.substitute_bytes(value)
        permuted = self.permute_bits(substituted)
        result = permuted ^ round_key
        
        return result
    
    def inverse_round_function(self, value, round_key):
        unmixed = value ^ round_key
        unpermuted = self.permute_bits(unmixed, inverse=True)
        unsubstituted = self.substitute_bytes(unpermuted, inverse=True)
        
        return unsubstituted
    
    def encrypt_block(self, plaintext):
        if len(plaintext) != 4:
            raise ValueError("Plaintext block must be 4 bytes (32 bits)")
        
        block = int.from_bytes(plaintext, byteorder='big')
        state = block ^ self.round_keys[0]
        
        for i in range(1, self.rounds):
            state = self.round_function(state, self.round_keys[i])
        
        return state.to_bytes(4, byteorder='big')
    
    def decrypt_block(self, ciphertext):
        if len(ciphertext) != 4:
            raise ValueError("Ciphertext block must be 4 bytes (32 bits)")
        
        block = int.from_bytes(ciphertext, byteorder='big')
        
        state = block
        for i in range(self.rounds - 1, 0, -1):
            state = self.inverse_round_function(state, self.round_keys[i])
        
        state = state ^ self.round_keys[0]
        
        return state.to_bytes(4, byteorder='big')
    
    def encrypt(self, plaintext):
        padding_length = 4 - (len(plaintext) % 4)
        if padding_length < 4:
            padded_plaintext = plaintext + bytes([padding_length]) * padding_length
        else:
            padded_plaintext = plaintext
        
        ciphertext = bytearray()
        for i in range(0, len(padded_plaintext), 4):
            block = padded_plaintext[i:i+4]
            encrypted_block = self.encrypt_block(block)
            ciphertext.extend(encrypted_block)
            
        return bytes(ciphertext)
    
    def decrypt(self, ciphertext):
        if len(ciphertext) % 4 != 0:
            raise ValueError("Ciphertext length must be a multiple of 4 bytes")
        
        plaintext = bytearray()
        for i in range(0, len(ciphertext), 4):
            block = ciphertext[i:i+4]
            decrypted_block = self.decrypt_block(block)
            plaintext.extend(decrypted_block)
        
        padding_length = plaintext[-1]
        if padding_length < 4:
            for i in range(1, padding_length + 1):
                if plaintext[-i] != padding_length:
                    return bytes(plaintext)
            return bytes(plaintext[:-padding_length])
        else:
            return bytes(plaintext)


def main():
    key = b'KEY!'  # 4-byte (32-bit) key
    cipher = Cipher32Bit(key)
    
    message = b'This is a secret message that will be encrypted!'
    
    encrypted = cipher.encrypt(message)
    print(f"Original message: {message.decode('utf-8')}")
    print(f"Encrypted (hex): {encrypted.hex()}")
    
    decrypted = cipher.decrypt(encrypted)
    print(f"Decrypted: {decrypted.decode('utf-8')}")
    
    assert decrypted == message, "Decryption failed! Original and decrypted messages don't match."
    print("Encryption and decryption successful!")
    
    test_messages = [
        b'A',
        b'AB',
        b'ABC',
        b'ABCD',
        b'ABCDE',
        b'A' * 15,
        b'A' * 16
    ]
    
    print("\nTesting with different message sizes:")
    for i, msg in enumerate(test_messages):
        enc = cipher.encrypt(msg)
        dec = cipher.decrypt(enc)
        print(f"Test {i+1} ({len(msg)} bytes): {'Success' if dec == msg else 'Failed'}")


if __name__ == "__main__":
    main()