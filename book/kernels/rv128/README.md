RV128I hart image. Little-endian 32-bit instruction parcels. XLEN 128. No vector parcel, so the filename records vlen0. sew64 is that same filename slot: ELEN stays 64 if a vector unit is attached later, and SEW 128 is not encoded. lq, sq, ldu, and addid follow QEMU's insn32.decode. addi is the XLEN add. slli by 64 uses the low 7 bits of the I-immediate. No host frame. These files are not ROMs.

- `quad.sew64.lmul1.tu.mu.vlen0.xlen128.le.bin`
- `add128.sew64.lmul1.tu.mu.vlen0.xlen128.le.bin`
- `addd16.sew64.lmul1.tu.mu.vlen0.xlen128.le.bin`
- `zext64.sew64.lmul1.tu.mu.vlen0.xlen128.le.bin`
- `sh64.sew64.lmul1.tu.mu.vlen0.xlen128.le.bin`
