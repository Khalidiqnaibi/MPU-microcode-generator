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

    def layout_layer(self, layer , weight_tensor):
        