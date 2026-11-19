import struct
import numpy as np

NOP = 0x00
LOAD_SDRAM = 0x01
MAC_EXEC = 0x02
APPLY_SCALE = 0x03
STORE_SDRAM = 0x04
JUMP = 0x05
HALT = 0x06
LIGHT = 0xD
OVERCLOCK = 0xE

class MPU_microcode_generator:
    def __init__(self):
        self.instructions = []
        self.current_sdram_offset = 0x000000

    def pack_instruction(self,opcode,sdram_addr,burst_len,dest_addr,flags):
        word = (
            ((opcode & 0xF)) |
            ((sdram_addr & 0xFFFFFF) << 4) |
            ((burst_len & 0xFFFF) << 28) |
            ((dest_addr & 0xFFFF) << 44) |
            ((flags & 0xF) << 60)
        )

        return word

    def layout_layer(self, layer_name , weight_tensor):
        print(f"Compiling layer {layer_name} with weight tensor shape {weight_tensor.shape}")
        out_feature,in_feature = weight_tensor.shape

        block_size = 64
        num_blocks = in_feature // block_size

        layer_start_addr = self.current_sdram_offset

        for out_idx in range(out_feature):
            for block_idx in range(num_blocks):
                addr = self.current_sdram_offset

                opcode = MAC_EXEC
                burst_len = block_size // 4  # Assuming each weight is 4 bytes
                dest_addr = out_idx

                flags = 0x1 if block_idx == 0 else 0x0

                instruction = self.pack_instruction(
                    opcode,
                    addr,
                    burst_len,
                    dest_addr,
                    flags
                )

                self.instructions.append(instruction)
                self.current_sdram_offset += burst_len
        return layer_start_addr

    

