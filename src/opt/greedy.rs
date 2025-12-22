/// BitMask with arbitrary precision using Vec<u128>
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitMask {
    /// Stores bits in little-endian order (bits[0] contains bits 0-127)
    bits: Vec<u128>,
}

impl BitMask {
    /// Create a new empty bitmask
    pub fn new() -> Self {
        BitMask { bits: vec![0] }
    }

    /// Create a new bitmask with the specified capacity (in number of bits)
    pub fn with_capacity(capacity_bits: u32) -> Self {
        let num_blocks = ((capacity_bits + 127) / 128) as usize;
        let mut bits = Vec::with_capacity(num_blocks);
        bits.push(0); // Start with at least one block
        BitMask { bits }
    }

    /// Create a bitmask from a single u128 value
    pub fn from_u128(value: u128) -> Self {
        BitMask { bits: vec![value] }
    }

    /// Create a mask with `count` consecutive ones starting from bit 0
    pub fn ones(count: u32) -> Self {
        if count == 0 {
            return BitMask::new();
        }

        let full_blocks = count / 128;
        let remaining = count % 128;

        let mut bits = vec![u128::MAX; full_blocks as usize];
        if remaining > 0 {
            bits.push((1u128 << remaining) - 1);
        }

        BitMask { bits }
    }

    /// Left shift operation
    pub fn shl(&self, shift: u32) -> Self {
        if shift == 0 {
            return self.clone();
        }

        let block_shift = (shift / 128) as usize;
        let bit_shift = shift % 128;

        if bit_shift == 0 {
            // Simple block shift
            let mut result = vec![0u128; block_shift];
            result.extend_from_slice(&self.bits);
            return BitMask { bits: result };
        }

        // Complex shift with bit spillover
        let mut result = vec![0u128; block_shift + self.bits.len() + 1];
        let right_shift = 128 - bit_shift;

        for (i, &block) in self.bits.iter().enumerate() {
            result[i + block_shift] |= block << bit_shift;
            if i + block_shift + 1 < result.len() {
                result[i + block_shift + 1] |= block >> right_shift;
            }
        }

        // Remove trailing zeros
        while result.len() > 1 && result.last() == Some(&0) {
            result.pop();
        }

        BitMask { bits: result }
    }

    /// Right shift operation
    pub fn shr(&self, shift: u32) -> Self {
        if shift == 0 {
            return self.clone();
        }

        let block_shift = (shift / 128) as usize;
        if block_shift >= self.bits.len() {
            return BitMask::new();
        }

        let bit_shift = shift % 128;

        if bit_shift == 0 {
            // Simple block shift
            return BitMask {
                bits: self.bits[block_shift..].to_vec(),
            };
        }

        // Complex shift with bit spillover
        let mut result = vec![0u128; self.bits.len() - block_shift];
        let left_shift = 128 - bit_shift;

        for i in 0..result.len() {
            result[i] = self.bits[i + block_shift] >> bit_shift;
            if i + block_shift + 1 < self.bits.len() {
                result[i] |= self.bits[i + block_shift + 1] << left_shift;
            }
        }

        // Remove trailing zeros
        while result.len() > 1 && result.last() == Some(&0) {
            result.pop();
        }

        BitMask { bits: result }
    }

    /// Left shift operation (mutating)
    pub fn shl_assign(&mut self, shift: u32) {
        if shift == 0 {
            return;
        }

        let block_shift = (shift / 128) as usize;
        let bit_shift = shift % 128;

        if bit_shift == 0 {
            // Simple block shift - insert zeros at the front
            self.bits
                .splice(0..0, std::iter::repeat(0).take(block_shift));
            return;
        }

        // Complex shift with bit spillover
        let right_shift = 128 - bit_shift;
        let mut carry = 0u128;

        // Shift existing blocks in place
        for block in &mut self.bits {
            let new_carry = *block >> right_shift;
            *block = (*block << bit_shift) | carry;
            carry = new_carry;
        }

        // Add carry as new block if non-zero
        if carry != 0 {
            self.bits.push(carry);
        }

        // Insert zero blocks at the front if needed
        if block_shift > 0 {
            self.bits
                .splice(0..0, std::iter::repeat(0).take(block_shift));
        }
    }

    /// Right shift operation (mutating)
    pub fn shr_assign(&mut self, shift: u32) {
        if shift == 0 {
            return;
        }

        let block_shift = (shift / 128) as usize;
        if block_shift >= self.bits.len() {
            self.bits.clear();
            self.bits.push(0);
            return;
        }

        let bit_shift = shift % 128;

        if bit_shift == 0 {
            // Simple block shift - remove blocks from the front
            self.bits.drain(0..block_shift);
            return;
        }

        // Remove full blocks first
        if block_shift > 0 {
            self.bits.drain(0..block_shift);
        }

        // Complex shift with bit spillover
        let left_shift = 128 - bit_shift;

        for i in 0..self.bits.len() {
            self.bits[i] >>= bit_shift;
            if i + 1 < self.bits.len() {
                self.bits[i] |= self.bits[i + 1] << left_shift;
            }
        }

        // Remove trailing zeros
        while self.bits.len() > 1 && self.bits.last() == Some(&0) {
            self.bits.pop();
        }
    }

    /// Bitwise AND operation
    pub fn and(&self, other: &BitMask) -> Self {
        let min_len = self.bits.len().min(other.bits.len());
        let mut result = vec![0u128; min_len];

        for i in 0..min_len {
            result[i] = self.bits[i] & other.bits[i];
        }

        // Remove trailing zeros
        while result.len() > 1 && result.last() == Some(&0) {
            result.pop();
        }

        BitMask { bits: result }
    }

    /// Bitwise XOR operation
    pub fn xor(&self, other: &BitMask) -> Self {
        let max_len = self.bits.len().max(other.bits.len());
        let mut result = vec![0u128; max_len];

        for i in 0..max_len {
            let a = self.bits.get(i).copied().unwrap_or(0);
            let b = other.bits.get(i).copied().unwrap_or(0);
            result[i] = a ^ b;
        }

        // Remove trailing zeros
        while result.len() > 1 && result.last() == Some(&0) {
            result.pop();
        }

        BitMask { bits: result }
    }

    /// Bitwise XOR operation (mutating)
    pub fn xor_assign(&mut self, other: &BitMask) {
        let max_len = self.bits.len().max(other.bits.len());

        // Extend self if needed
        if self.bits.len() < max_len {
            self.bits.resize(max_len, 0);
        }

        // XOR in place
        for i in 0..max_len {
            let b = other.bits.get(i).copied().unwrap_or(0);
            self.bits[i] ^= b;
        }

        // Remove trailing zeros
        while self.bits.len() > 1 && self.bits.last() == Some(&0) {
            self.bits.pop();
        }
    }

    /// Count trailing zeros
    pub fn trailing_zeros(&self) -> u32 {
        for (i, &block) in self.bits.iter().enumerate() {
            if block != 0 {
                return (i as u32) * 128 + block.trailing_zeros();
            }
        }
        // All zeros
        self.bits.len() as u32 * 128
    }

    /// Count trailing zeros starting from a specific bit offset (avoids allocating a shifted BitMask)
    pub fn trailing_zeros_from(&self, offset: u32) -> u32 {
        let block_index = (offset / 128) as usize;
        let bit_offset = offset % 128;

        if block_index >= self.bits.len() {
            return 0;
        }

        // Check first block with offset
        let first_block = self.bits[block_index] >> bit_offset;
        if first_block != 0 {
            return first_block.trailing_zeros();
        }

        let mut count = 128 - bit_offset;

        // Check remaining blocks
        for i in (block_index + 1)..self.bits.len() {
            if self.bits[i] != 0 {
                return count + self.bits[i].trailing_zeros();
            }
            count += 128;
        }

        count
    }

    /// Count trailing ones
    pub fn trailing_ones(&self) -> u32 {
        for (i, &block) in self.bits.iter().enumerate() {
            if block != u128::MAX {
                return (i as u32) * 128 + block.trailing_ones();
            }
        }
        // All ones
        self.bits.len() as u32 * 128
    }

    /// Count trailing ones starting from a specific bit offset (avoids allocating a shifted BitMask)
    pub fn trailing_ones_from(&self, offset: u32) -> u32 {
        let block_index = (offset / 128) as usize;
        let bit_offset = offset % 128;

        if block_index >= self.bits.len() {
            return 0;
        }

        // Check first block with offset
        let first_block = self.bits[block_index] >> bit_offset;
        let mask = u128::MAX >> bit_offset;

        if first_block != mask {
            return first_block.trailing_ones();
        }

        let mut count = 128 - bit_offset;

        // Check remaining blocks
        for i in (block_index + 1)..self.bits.len() {
            if self.bits[i] != u128::MAX {
                return count + self.bits[i].trailing_ones();
            }
            count += 128;
        }

        count
    }

    /// Check if all bits are zero
    pub fn is_zero(&self) -> bool {
        self.bits.iter().all(|&b| b == 0)
    }

    /// Set a specific bit to 1 (mutating)
    pub fn set_bit(&mut self, bit_index: u32) {
        let block_index = (bit_index / 128) as usize;
        let bit_offset = bit_index % 128;

        // Extend the vector if necessary
        if block_index >= self.bits.len() {
            self.bits.resize(block_index + 1, 0);
        }

        self.bits[block_index] |= 1u128 << bit_offset;
    }

    /// Check if the nth bit is set (returns true if the bit is 1, false otherwise)
    pub fn nth(&self, bit_index: u32) -> bool {
        let block_index = (bit_index / 128) as usize;
        let bit_offset = bit_index % 128;

        if block_index >= self.bits.len() {
            return false;
        }

        (self.bits[block_index] & (1u128 << bit_offset)) != 0
    }
}

impl std::ops::BitAnd for &BitMask {
    type Output = BitMask;

    fn bitand(self, rhs: &BitMask) -> BitMask {
        self.and(rhs)
    }
}

impl std::ops::BitXor for &BitMask {
    type Output = BitMask;

    fn bitxor(self, rhs: &BitMask) -> BitMask {
        self.xor(rhs)
    }
}

impl std::ops::BitOr for &BitMask {
    type Output = BitMask;

    fn bitor(self, rhs: &BitMask) -> BitMask {
        let max_len = self.bits.len().max(rhs.bits.len());
        let mut result = vec![0u128; max_len];

        for i in 0..max_len {
            let a = self.bits.get(i).copied().unwrap_or(0);
            let b = rhs.bits.get(i).copied().unwrap_or(0);
            result[i] = a | b;
        }

        // Remove trailing zeros
        while result.len() > 1 && result.last() == Some(&0) {
            result.pop();
        }

        BitMask { bits: result }
    }
}

impl std::ops::Shl<u32> for &BitMask {
    type Output = BitMask;

    fn shl(self, rhs: u32) -> BitMask {
        self.shl(rhs)
    }
}

impl std::ops::Shr<u32> for &BitMask {
    type Output = BitMask;

    fn shr(self, rhs: u32) -> BitMask {
        self.shr(rhs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GreedyQuad {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Generate quads from a binary plane represented as BitMask bitmasks
/// Each BitMask in the vector represents a row, with each bit representing a column
/// max_size limits the maximum width/height of quads (in pixels before scaling)
pub fn greedy_mesh_binary_plane(
    mut data: Vec<BitMask>,
    max_x: u32,
    max_y: u32,
    max_size: u32,
) -> Vec<GreedyQuad> {
    let mut greedy_quads = vec![];

    for row in 0..data.len().min(max_x as usize) {
        let mut y = 0;
        while y < max_y {
            // Find first solid bit, skip "air/zeros" - use optimized version
            let trailing_zeros = data[row].trailing_zeros_from(y).min(max_y - y);
            y += trailing_zeros;

            if y >= max_y {
                // Reached top
                break;
            }

            // Count consecutive ones (height) - use optimized version
            // Limit height to max_size
            let h = data[row].trailing_ones_from(y).min(max_y - y).min(max_size);

            if h == 0 {
                // No ones found, skip this position
                y += 1;
                continue;
            }

            // Create mask for the height once
            let h_as_mask = BitMask::ones(h);
            let mask = h_as_mask.shl(y);

            // Grow horizontally (limit to max_size)
            let mut w = 1;
            while row + w < data.len() && w < max_size as usize {
                // Fetch bits spanning height in the next row
                let next_row_bits = data[row + w].shr(y).and(&h_as_mask);

                if next_row_bits != h_as_mask {
                    break; // Can no longer expand
                }

                // Clear the bits we expanded into by XORing with the mask (mutating)
                data[row + w].xor_assign(&mask);

                w += 1;
            }

            greedy_quads.push(GreedyQuad {
                x: row as u32,
                y,
                w: w as u32,
                h,
            });

            // Clear the bits from the current row (mutating)
            data[row].xor_assign(&mask);

            y += h;
        }
    }

    greedy_quads
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitmask_basic() {
        let mask = BitMask::from_u128(0b1111);
        assert_eq!(mask.trailing_zeros(), 0);
        assert_eq!(mask.trailing_ones(), 4);
    }

    #[test]
    fn test_bitmask_shift_left() {
        let mask = BitMask::from_u128(0b1111);
        let shifted = mask.shl(2);
        // After shifting left by 2: 0b1111 -> 0b111100
        // This has 2 trailing zeros, not trailing ones
        assert_eq!(shifted.trailing_zeros(), 2);
        // To count the ones, we need to skip the zeros first
        let after_zeros = shifted.shr(2);
        assert_eq!(after_zeros.trailing_ones(), 4);
    }

    #[test]
    fn test_bitmask_shift_right() {
        let mask = BitMask::from_u128(0b111100);
        let shifted = mask.shr(2);
        assert_eq!(shifted.trailing_zeros(), 0);
        assert_eq!(shifted.trailing_ones(), 4);
    }

    #[test]
    fn test_bitmask_ones() {
        let mask = BitMask::ones(5);
        assert_eq!(mask.trailing_zeros(), 0);
        assert_eq!(mask.trailing_ones(), 5);
    }

    #[test]
    fn test_bitmask_and() {
        let a = BitMask::from_u128(0b1111);
        let b = BitMask::from_u128(0b1100);
        let result = a.and(&b);
        assert_eq!(result, BitMask::from_u128(0b1100));
    }

    #[test]
    fn test_bitmask_xor() {
        let a = BitMask::from_u128(0b1111);
        let b = BitMask::from_u128(0b1100);
        let result = a.xor(&b);
        assert_eq!(result, BitMask::from_u128(0b0011));
    }

    #[test]
    fn test_greedy_mesh_simple() {
        // Create a simple 2x2 solid block
        // Row 0: 0b11 (bits 0-1 set)
        // Row 1: 0b11 (bits 0-1 set)
        let data = vec![BitMask::from_u128(0b11), BitMask::from_u128(0b11)];

        let quads = greedy_mesh_binary_plane(data, 2, 2, 1000);

        // Should produce a single 2x2 quad
        assert_eq!(quads.len(), 1);
        assert_eq!(
            quads[0],
            GreedyQuad {
                x: 0,
                y: 0,
                w: 2,
                h: 2
            }
        );
    }

    #[test]
    fn test_greedy_mesh_separate_quads() {
        // Create two separate quads
        // Row 0: 0b1100 (bits 2-3 set)
        // Row 1: 0b0011 (bits 0-1 set)
        let data = vec![BitMask::from_u128(0b1100), BitMask::from_u128(0b0011)];

        let quads = greedy_mesh_binary_plane(data, 2, 4, 1000);

        // Should produce two separate quads
        assert_eq!(quads.len(), 2);
    }
}
