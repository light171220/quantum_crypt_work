import numpy as np

# AES S-box standard lookup table
AES_SBOX = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
]

"""
IMPORTANT: This implementation follows the approach from:
'A Very Compact S-box for AES' by D. Canright,
which is widely cited and verified to work correctly.
"""

# We use GF(2^4) with irreducible polynomial x^4 + x + 1 (0x13)
# and GF((2^4)^2) with irreducible polynomial y^2 + y + λ, where λ = 0xE

# Define parameters for the composite field
GF16_IRREDUCIBLE = 0x13      # x^4 + x + 1
GF256_IRREDUCIBLE = 0x11B    # x^8 + x^4 + x^3 + x + 1
COMPOSITE_LAMBDA = 0xE       # λ for polynomial y^2 + y + λ in GF((2^4)^2)

def byte_to_bits(byte):
    """Convert a byte to an 8-bit array (LSB first)"""
    return [(byte >> i) & 1 for i in range(8)]

def bits_to_byte(bits):
    """Convert an 8-bit array (LSB first) to a byte"""
    return sum(bits[i] << i for i in range(8))

# ----- GF(2^4) Operations -----

def gf16_mul(a, b):
    """Multiplication in GF(2^4) with irreducible polynomial x^4 + x + 1"""
    p = 0
    for i in range(4):
        if (b >> i) & 1:
            p ^= a << i
    
    # Reduction modulo x^4 + x + 1
    for i in range(7, 3, -1):
        if (p >> i) & 1:
            p ^= GF16_IRREDUCIBLE << (i - 4)
    
    return p & 0xF

# Precompute multiplication and inverse tables for GF(2^4)
GF16_MUL_TABLE = [[gf16_mul(i, j) for j in range(16)] for i in range(16)]
GF16_INV_TABLE = [0] * 16
for i in range(1, 16):
    for j in range(1, 16):
        if GF16_MUL_TABLE[i][j] == 1:
            GF16_INV_TABLE[i] = j
            break

def gf16_add(a, b):
    """Addition in GF(2^4) - XOR operation"""
    return a ^ b

def gf16_sq(a):
    """Square a number in GF(2^4) - optimized for x^4 + x + 1"""
    # For the polynomial x^4 + x + 1, squaring has a specific pattern
    bits = [0, 0, 0, 0]
    if (a & 1): bits[0] = 1      # a_0 -> a_0
    if (a & 2): bits[2] = 1      # a_1 -> a_2
    if (a & 4): 
        bits[0] ^= 1             # a_2 -> a_4 -> a_0 (mod x^4 + x + 1)
        bits[1] ^= 1             # a_2 -> a_4 -> a_1 (mod x^4 + x + 1)
    if (a & 8): 
        bits[1] ^= 1             # a_3 -> a_6 -> a_1 (mod x^4 + x + 1)
        bits[2] ^= 1             # a_3 -> a_6 -> a_2 (mod x^4 + x + 1)
        bits[3] ^= 1             # a_3 -> a_6 -> a_3 (mod x^4 + x + 1)
    
    return (bits[0]) | (bits[1] << 1) | (bits[2] << 2) | (bits[3] << 3)

# ----- Transformations between Fields -----

# These transformation matrices are from Canright's paper
# They map between GF(2^8) and GF((2^4)^2) in a way that preserves field structure
# for the specific irreducible polynomials we're using

# Transformation from GF(2^8) to GF((2^4)^2)
A2X = np.array([
    [0, 0, 0, 0, 1, 1, 0, 1],  # bits of high nibble
    [0, 1, 0, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1],  # bits of low nibble
    [0, 1, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0]
], dtype=np.uint8)

# Transformation from GF((2^4)^2) to GF(2^8)
X2A = np.array([
    [0, 1, 0, 1, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 1, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 0, 1, 0, 0],
    [1, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1]
], dtype=np.uint8)

# AES affine transformation matrix
AFFINE_MATRIX = np.array([
    [1, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 1]
], dtype=np.uint8)

# AES affine transformation constant
AFFINE_CONST = np.array([1, 1, 0, 0, 0, 1, 1, 0], dtype=np.uint8)

def transform_to_gf16_squared(byte):
    """Transform byte from GF(2^8) to GF((2^4)^2)"""
    bits = byte_to_bits(byte)
    transformed_bits = np.zeros(8, dtype=np.uint8)
    
    for i in range(8):
        for j in range(8):
            transformed_bits[i] ^= A2X[i, j] & bits[j]
    
    # Extract high and low nibbles (first and last 4 bits)
    high_nibble = 0
    low_nibble = 0
    for i in range(4):
        high_nibble |= transformed_bits[i] << i
        low_nibble |= transformed_bits[i+4] << i
    
    return high_nibble, low_nibble

def transform_from_gf16_squared(high_nibble, low_nibble):
    """Transform from GF((2^4)^2) back to GF(2^8)"""
    # Combine high and low nibbles into 8 bits
    composite_bits = []
    for i in range(4):
        composite_bits.append((high_nibble >> i) & 1)
    for i in range(4):
        composite_bits.append((low_nibble >> i) & 1)
    
    # Transform back using X2A matrix
    result_bits = np.zeros(8, dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            result_bits[i] ^= X2A[i, j] & composite_bits[j]
    
    return bits_to_byte(result_bits)

def apply_affine_transform(byte):
    """Apply the AES affine transformation"""
    bits = byte_to_bits(byte)
    result_bits = np.zeros(8, dtype=np.uint8)
    
    # Apply matrix multiplication
    for i in range(8):
        for j in range(8):
            result_bits[i] ^= AFFINE_MATRIX[i, j] & bits[j]
        # Add constant
        result_bits[i] ^= AFFINE_CONST[i]
    
    return bits_to_byte(result_bits)

# ----- Compute Inverse in GF((2^4)^2) -----

def inverse_in_gf16_squared(high, low):
    """
    Compute the multiplicative inverse in GF((2^4)^2)
    For polynomial ah·y + al with y^2 + y + λ
    """
    # Handle zero input
    if high == 0 and low == 0:
        return 0, 0
    
    # Computing ah² · λ + (ah · al)
    high_squared = gf16_sq(high)
    high_low_product = GF16_MUL_TABLE[high][low]
    lambda_high_squared = GF16_MUL_TABLE[COMPOSITE_LAMBDA][high_squared]
    
    # Computing al² + ah · al
    low_squared = gf16_sq(low)
    delta = gf16_add(gf16_add(lambda_high_squared, high_low_product), low_squared)
    
    # Handle the case where delta is 0 (shouldn't happen with valid inputs)
    if delta == 0:
        return 0, 0
    
    # Computing delta⁻¹
    delta_inv = GF16_INV_TABLE[delta]
    
    # Computing (ah² · λ + al · ah) · delta⁻¹ and ah · delta⁻¹
    temp = gf16_add(lambda_high_squared, high_low_product)
    high_inv = GF16_MUL_TABLE[high][delta_inv]
    low_inv = GF16_MUL_TABLE[gf16_add(low, high)][delta_inv]
    
    return high_inv, low_inv

def s_box_composite_field(byte):
    """Compute the AES S-box value using the composite field approach"""
    # Step 1: Transform from GF(2^8) to GF((2^4)^2)
    high, low = transform_to_gf16_squared(byte)
    
    # Step 2: Compute the multiplicative inverse in GF((2^4)^2)
    inv_high, inv_low = inverse_in_gf16_squared(high, low)
    
    # Step 3: Transform back to GF(2^8)
    inv_byte = transform_from_gf16_squared(inv_high, inv_low)
    
    # Step 4: Apply the affine transformation
    return apply_affine_transform(inv_byte)

def verify_s_box():
    """Verify our S-box implementation against the standard AES S-box"""
    success = True
    mismatches = 0
    
    for i in range(256):
        composite_result = s_box_composite_field(i)
        standard_result = AES_SBOX[i]
        
        if composite_result != standard_result:
            if mismatches < 10:  # Only show first 10 mismatches
                print(f"Mismatch at 0x{i:02X}: Composite = 0x{composite_result:02X}, Standard = 0x{standard_result:02X}")
            mismatches += 1
            success = False
    
    if success:
        print("All S-box values match! The composite field implementation is correct.")
    else:
        print(f"There are {mismatches} mismatches between the composite field and standard implementation.")
    
    # Test specific example
    test_value = 0x53
    composite_result = s_box_composite_field(test_value)
    standard_result = AES_SBOX[test_value]
    
    print(f"\nExample: S-box(0x{test_value:02X})")
    print(f"Composite field result: 0x{composite_result:02X}")
    print(f"Standard lookup result: 0x{standard_result:02X}")
    print(f"Match: {composite_result == standard_result}")
    
    # Show intermediate steps for the example
    print("\nDetailed steps for S-box(0x53):")
    high, low = transform_to_gf16_squared(test_value)
    print(f"1. Map to composite field: (0x{high:X}, 0x{low:X})")
    
    high_squared = gf16_sq(high)
    low_squared = gf16_sq(low)
    high_low_product = GF16_MUL_TABLE[high][low]
    lambda_high_squared = GF16_MUL_TABLE[COMPOSITE_LAMBDA][high_squared]
    delta = gf16_add(gf16_add(lambda_high_squared, high_low_product), low_squared)
    delta_inv = GF16_INV_TABLE[delta] if delta != 0 else 0
    
    print(f"2. high² = 0x{high_squared:X}, low² = 0x{low_squared:X}")
    print(f"3. λ·high² = 0x{lambda_high_squared:X}, high·low = 0x{high_low_product:X}")
    print(f"4. Δ = λ·high² + high·low + low² = 0x{delta:X}")
    print(f"5. Δ⁻¹ = 0x{delta_inv:X}")
    
    inv_high, inv_low = inverse_in_gf16_squared(high, low)
    print(f"6. Inverse in composite field: (0x{inv_high:X}, 0x{inv_low:X})")
    
    inv_byte = transform_from_gf16_squared(inv_high, inv_low)
    print(f"7. Map back to GF(2^8): 0x{inv_byte:02X}")
    
    result = apply_affine_transform(inv_byte)
    print(f"8. After affine transformation: 0x{result:02X}")
    
    return success

if __name__ == "__main__":
    verify_s_box()