RV64 hart image, V or Zve64d. Little-endian 32-bit instruction parcels (the instruction word is still 32 bits). SEW as named, LMUL 1, tail undisturbed, mask undisturbed. VLEN 128 is the floor. XLEN 64. No host frame in this edition; the BX-32 COP3 framing is the pattern, with the low 64 bits of a 64-bit GPR as the address.

- `sew64.sew64.lmul1.tu.mu.vlen128.xlen64.le.bin`
- `fma64.sew64.lmul1.tu.mu.vlen128.xlen64.le.bin`
- `gemm64.sew64.lmul1.tu.mu.vlen128.xlen64.le.bin`
- `index64.sew64.lmul1.tu.mu.vlen128.xlen64.le.bin`
- `scal64.sew64.lmul1.tu.mu.vlen128.xlen64.le.bin`
- `copy64.sew64.lmul1.tu.mu.vlen128.xlen64.le.bin`
- `axpy64.sew64.lmul1.tu.mu.vlen128.xlen64.le.bin`
- `dot64.sew64.lmul1.tu.mu.vlen128.xlen64.le.bin`
- `nrm264.sew64.lmul1.tu.mu.vlen128.xlen64.le.bin`
- `asum64.sew64.lmul1.tu.mu.vlen128.xlen64.le.bin`
- `swap64.sew64.lmul1.tu.mu.vlen128.xlen64.le.bin`
- `rot64.sew64.lmul1.tu.mu.vlen128.xlen64.le.bin`
- `gemv64.sew64.lmul1.tu.mu.vlen128.xlen64.le.bin`
- `ger64.sew64.lmul1.tu.mu.vlen128.xlen64.le.bin`
- `iamax64.sew64.lmul1.tu.mu.vlen128.xlen64.le.bin`
