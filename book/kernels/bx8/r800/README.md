Harness image for an R800 softcore. Each RV parcel, scalar or vector, is preceded by the escape ED 00. Bytes are little-endian. A ROM escapes only OP-V and writes the branch and the address update in R800 instructions. The rs1 field here is still the RV number. Re-assemble it as the host index before you put it in a ROM. Map: 0 BC, 1 DE, 2 HL, 3 IX, 4 IY, 5 SP.

- `fill8.sew8.lmul1.tu.mu.vlen32.xlen32.le.bin`
- `copy8.sew8.lmul1.tu.mu.vlen32.xlen32.le.bin`
- `mask8.sew8.lmul1.tu.mu.vlen32.xlen32.le.bin`
- `add8.sew8.lmul1.tu.mu.vlen32.xlen32.le.bin`
