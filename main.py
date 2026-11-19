import struct
import numpy as np

NOP = 0x0
LOAD_SDRAM = 0x1
MAC_EXEC = 0x2
APPLY_SCALE = 0x3
STORE_SDRAM = 0x4
JUMP = 0x5
HALT = 0x6
LIGHT = 0xD
OVERCLOCK = 0xE

FLAG_NONE = 0x0
FLAG_CLEAR_ACCUM = 0x1
FLAG_HAS_SCALE = 0x2

class MPU_microcode_generator:
    def __init__(self,bus_width_bits: int =32,bits_per_weight: int =1):
        assert bus_width_bits % 8 == 0
        assert bits_per_weight in (1, 2)
        self.instructions:list[int] = []
        self.current_sdram_offset = 0x000000
        self.bus_width_bytes: int = bus_width_bits // 8
        self.bits_per_weight: int = bits_per_weight

    def pack_instruction(
            self,opcode:int,sdram_addr:int,burst_len:int,dest_addr:int,flags:int
            ) -> int:
        word = (
            ((opcode & 0xF)) |
            ((sdram_addr & 0xFFFFFF) << 4) |
            ((burst_len & 0xFFFF) << 28) |
            ((dest_addr & 0xFFFF) << 44) |
            ((flags & 0xF) << 60)
        )

        return word

    def pack_1bit_weights(self,weights_1d: np.ndarray) -> bytes:
        bits = (weights_1d.ravel()> 0).astype(np.uint8)
        rem = (-len(bits))% 8
        if rem:
            bits = np.pad(bits,(0,rem),constrant_values=0)
        packed = np.packbits(bits,bitorder="big")
        return packed.tobytes()

    def layout_layer(
        self,
        layer_name: str,
        weight_tensor: np.ndarray,
        out_sram_base: int = 0,
        block_weights: int = 64,
    ) -> tuple[int, bytearray]:
        
        if weight_tensor.ndim != 2:
            raise ValueError(f"Expected 2D tensor for {layer_name}, got {weight_tensor.shape}")

        out_features, in_features = weight_tensor.shape
        padded_in = ((in_features + block_weights - 1) // block_weights) * block_weights
        if padded_in != in_features:
            weight_tensor = np.pad(
                weight_tensor, ((0, 0), (0, padded_in - in_features)), mode="constant"
            )

        num_blocks = padded_in // block_weights
        bytes_per_block = (block_weights * self.bits_per_weight + 7) // 8
        burst_beats = max(1, (bytes_per_block + self.bus_width_bytes - 1) // self.bus_width_bytes)

        layer_start_addr = self.current_sdram_byte_offset
        weight_blob = bytearray()

        print(
            f"Compiling '{layer_name}': shape={weight_tensor.shape}, "
            f"blocks/row={num_blocks}, burst_beats={burst_beats}, start=0x{layer_start_addr:06X}"
        )

        for out_idx in range(out_features):
            for blk_idx in range(num_blocks):
                w_slice = weight_tensor[out_idx, blk_idx * block_weights : (blk_idx + 1) * block_weights]
                packed_slice = self.pack_1bit_weights(w_slice)
                weight_blob.extend(packed_slice)

                addr = self.current_sdram_byte_offset
                flags = FLAG_CLEAR_ACCUM if blk_idx == 0 else FLAG_NONE

                self.instructions.append(
                    self.pack_instruction(
                        opcode=MAC_EXEC,
                        sdram_addr=addr,
                        burst_beats=burst_beats,
                        dest_addr=(out_sram_base + out_idx) & 0xFFFF,
                        flags=flags,
                    )
                )
                self.current_sdram_byte_offset += len(packed_slice)

            self.instructions.append(
                self.pack_instruction(
                    opcode=APPLY_SCALE,
                    sdram_addr=0,
                    burst_beats=1,
                    dest_addr=(out_sram_base + out_idx) & 0xFFFF,
                    flags=FLAG_HAS_SCALE,
                )
            )
            self.instructions.append(
                self.pack_instruction(
                    opcode=STORE_SDRAM,
                    sdram_addr=0,
                    burst_beats=1,
                    dest_addr=(out_sram_base + out_idx) & 0xFFFF,
                    flags=FLAG_NONE,
                )
            )

        return layer_start_addr, weight_blob

    def generate_binary(self, ucode_filename: str = "firmware.ucode", weights_filename: str = "weights.bin", weights_blob: bytes = b""):
        self.instructions.append(self.pack_instruction(HALT, 0, 0, 0, 0))

        with open(ucode_filename, "wb") as f:
            for instruction in self.instructions:
                f.write(struct.pack("<Q", instruction))

        with open(weights_filename, "wb") as wf:
            wf.write(weights_blob)

        print(
            f"Wrote {len(self.instructions)} instructions -> {ucode_filename} "
            f"and {len(weights_blob)} weight bytes -> {weights_filename}"
        )


if __name__ == "__main__":
    microcode_gen = MPU_microcode_generator()




