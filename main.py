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

    def layout_layer(self, layer_name: str, weight_tensor: np.ndarray) -> bytes:
        print(f"Compiling layer {layer_name} with weight tensor shape {weight_tensor.shape}")
        out_feature,in_feature = weight_tensor.shape

        block_size = self.bus_width_bits
        num_blocks = in_feature // block_size

        layer_start_addr = self.current_sdram_offset

        for out_idx in range(out_feature):
            for block_idx in range(num_blocks):
                addr = self.current_sdram_offset

                opcode = MAC_EXEC
                burst_len = block_size // 4  # Assuming each weight is 4 bytes
                dest_addr = out_idx

                flags = FLAG_CLEAR_ACCUM if block_idx == 0 else FLAG_NONE

                instruction = self.pack_instruction(
                    opcode,
                    addr,
                    burst_len,
                    dest_addr,
                    flags
                )

                self.instructions.append(instruction)
                self.current_sdram_offset += burst_len

                light_instruction = self.pack_instruction(LIGHT, 0, 0, 0, 0)
                self.instructions.append(light_instruction)
        return layer_start_addr

    def generate_binary(self, filename: str ="firmware.ucode") -> None:

        halt_instruction = self.pack_instruction(HALT, 0, 0, 0, 0)
        self.instructions.append(halt_instruction)

        with open(filename, "wb") as f:
            for instruction in self.instructions:
                f.write(struct.pack("<Q", instruction))
        print(
            f"Microcode binary written to {filename} ",
            f"total instructions: {len(self.instructions)}"
        )


if __name__ == "__main__":
    microcode_gen = MPU_microcode_generator()




