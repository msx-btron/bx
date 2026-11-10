#!/usr/bin/env python3
"""Encode the BX acceptance kernels as raw RV parcels and host-framed images.

The vector encodings follow the layouts printed in the machine chapters
(OP-V, LOAD-FP, STORE-FP) and the RVV 1.0 field assignments those layouts name.
Scalar words are RV32I. VLEN is not a field in the instruction; the filename
records the profile the acceptance test is aimed at.

Host files are harness images: the escape is inserted before every RV parcel,
including scalar words. A ROM escapes only OP-V and writes branches in the
host ISA. See the folder README.
"""

import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parent

REG = {
    "zero": 0, "ra": 1, "t0": 5, "t1": 6,
    "a0": 10, "a1": 11, "a2": 12, "a3": 13,
}
FREG = {"ft0": 0, "ft1": 1}

SEW = {8: 0, 16: 1, 32: 2, 64: 3}
WIDTH = {8: 0, 16: 5, 32: 6, 64: 7}  # load/store W field
OPIVV, OPFVV, OPMVV, OPIVI, OPIVX, OPFVF = 0, 1, 2, 3, 4, 5
LOAD, STORE, OPV = 0x07, 0x27, 0x57


def opv(funct6, vm, vs2, vs1, funct3, vd):
    return ((funct6 & 0x3F) << 26) | ((vm & 1) << 25) | ((vs2 & 0x1F) << 20) | (
        (vs1 & 0x1F) << 15) | ((funct3 & 7) << 12) | ((vd & 0x1F) << 7) | OPV


def vsetvli(rd, rs1, sew):
    """e*, m1, tu, mu. zimm[2:0] holds vsew. Bit 31 stays clear."""
    zimm = SEW[sew]
    return (zimm << 20) | (REG[rs1] << 15) | (7 << 12) | (REG[rd] << 7) | OPV


def vload(width, rs1, vd, vm=1, mop=0, rs2=0, opcode=LOAD):
    return ((mop & 3) << 26) | ((vm & 1) << 25) | ((rs2 & 0x1F) << 20) | (
        REG[rs1] << 15) | (WIDTH[width] << 12) | ((vd & 0x1F) << 7) | opcode


def rtype(funct7, rs2, rs1, funct3, rd, opcode=0x33):
    return (funct7 << 25) | (REG[rs2] << 20) | (REG[rs1] << 15) | (
        funct3 << 12) | (REG[rd] << 7) | opcode


def itype(imm, rs1, funct3, rd, opcode=0x13):
    return ((imm & 0xFFF) << 20) | (REG[rs1] << 15) | (funct3 << 12) | (
        REG[rd] << 7) | opcode


def btype(funct3, rs1, rs2, imm):
    if imm & 1 or imm < -4096 or imm > 4094:
        raise ValueError(f"bad branch displacement {imm}")
    imm12 = (imm >> 12) & 1
    imm11 = (imm >> 11) & 1
    imm10_5 = (imm >> 5) & 0x3F
    imm4_1 = (imm >> 1) & 0xF
    return (imm12 << 31) | (imm10_5 << 25) | (REG[rs2] << 20) | (
        REG[rs1] << 15) | (funct3 << 12) | (imm4_1 << 8) | (imm11 << 7) | 0x63


def encode(op, pc, labels):
    k = op[0]
    if k == "vsetvli":
        return vsetvli(op[1], op[2], op[3])
    if k == "vmv.vx":
        # funct6 010111, OPIVX, vs2 = 0, vm = 1
        return opv(0b010111, 1, 0, REG[op[2]], OPIVX, int(op[1][1:]))
    if k == "vmv.vi":
        return opv(0b010111, 1, 0, op[2] & 0x1F, OPIVI, int(op[1][1:]))
    if k == "vmv.xs":
        # funct6 010000, OPMVV, vs1 = 0, vm = 1. rd sits in the vd field.
        return opv(0b010000, 1, int(op[2][1:]), 0, OPMVV, REG[op[1]])
    if k == "vle":
        return vload(op[1], op[3], int(op[2][1:]))
    if k == "vse":
        return vload(op[1], op[3], int(op[2][1:]), vm=0 if op[4:] == ("m",) else 1, opcode=STORE)
    if k == "vlse":
        return vload(op[1], op[3], int(op[2][1:]), mop=0b10, rs2=REG[op[4]])
    if k == "vluxei":
        return vload(op[1], op[3], int(op[2][1:]), mop=0b01, rs2=int(op[4][1:]))
    if k == "vsadd.vv":
        return opv(0b100001, 1, int(op[2][1:]), int(op[3][1:]), OPIVV, int(op[1][1:]))
    if k == "vsaddu.vv":
        # funct6 100000, the unsigned saturating add in the BX-16 map.
        return opv(0b100000, 1, int(op[2][1:]), int(op[3][1:]), OPIVV, int(op[1][1:]))
    if k == "vsub.vv":
        return opv(0b000010, 1, int(op[2][1:]), int(op[3][1:]), OPIVV, int(op[1][1:]))
    if k == "vmax.vv":
        return opv(0b000111, 1, int(op[2][1:]), int(op[3][1:]), OPIVV, int(op[1][1:]))
    if k == "vmaxu.vv":
        return opv(0b000110, 1, int(op[2][1:]), int(op[3][1:]), OPIVV, int(op[1][1:]))
    if k == "vmsltu.vx":
        return opv(0b011010, 1, int(op[2][1:]), REG[op[3]], OPIVX, int(op[1][1:]))
    if k == "vwmul.vv":
        # funct6 111011, OPMVV. Assembly is vd, vs2, vs1.
        return opv(0b111011, 1, int(op[2][1:]), int(op[3][1:]), OPMVV, int(op[1][1:]))
    if k == "vnclip.wi":
        return opv(0b101111, 1, int(op[2][1:]), op[3] & 0x1F, OPIVI, int(op[1][1:]))
    if k == "vredsum.vs":
        return opv(0b000000, 1, int(op[2][1:]), int(op[3][1:]), OPMVV, int(op[1][1:]))
    if k == "vfmacc.vv":
        # funct6 101100, OPFVV. Assembly is vd, vs1, vs2.
        return opv(0b101100, 1, int(op[3][1:]), int(op[2][1:]), OPFVV, int(op[1][1:]))
    if k == "vfmacc.vf":
        # Same funct6, OPFVF. Assembly is vd, fs1, vs2. rs1 is an F register.
        return opv(0b101100, 1, int(op[3][1:]), FREG[op[2]], OPFVF, int(op[1][1:]))
    if k == "vfmul.vv":
        return opv(0b100100, 1, int(op[2][1:]), int(op[3][1:]), OPFVV, int(op[1][1:]))
    if k == "vfmul.vf":
        return opv(0b100100, 1, int(op[2][1:]), FREG[op[3]], OPFVF, int(op[1][1:]))
    if k == "vfadd.vv":
        return opv(0b000000, 1, int(op[2][1:]), int(op[3][1:]), OPFVV, int(op[1][1:]))
    if k == "vfsub.vv":
        return opv(0b000010, 1, int(op[2][1:]), int(op[3][1:]), OPFVV, int(op[1][1:]))
    if k == "vfsgnjx.vv":
        return opv(0b001010, 1, int(op[2][1:]), int(op[3][1:]), OPFVV, int(op[1][1:]))
    if k == "vfredusum.vs":
        return opv(0b000001, 1, int(op[2][1:]), int(op[3][1:]), OPFVV, int(op[1][1:]))
    if k == "vfredmax.vs":
        return opv(0b000111, 1, int(op[2][1:]), int(op[3][1:]), OPFVV, int(op[1][1:]))
    if k == "vmfeq.vf":
        return opv(0b011000, 1, int(op[2][1:]), FREG[op[3]], OPFVF, int(op[1][1:]))
    if k == "vfmv.fs":
        # vfmv.f.s rd, vs2. funct6 010000, OPFVV, vs1 = 0. rd is an F register.
        return opv(0b010000, 1, int(op[2][1:]), 0, OPFVV, FREG[op[1]])
    if k == "vfirst":
        # vfirst.m rd, vs2. VWXUNARY0, vs1 field = 10001.
        return opv(0b010000, 1, int(op[2][1:]), 0b10001, OPMVV, REG[op[1]])
    if k in ("flw", "fld"):
        # Scalar LOAD-FP. funct3 010 is a word, 011 is a double.
        # The opcode byte matches a vector load; the width field does not.
        funct3 = 0b010 if k == "flw" else 0b011
        return ((op[3] & 0xFFF) << 20) | (REG[op[2]] << 15) | (funct3 << 12) | (
            FREG[op[1]] << 7) | LOAD
    if k in ("fsw", "fsd"):
        funct3 = 0b010 if k == "fsw" else 0b011
        imm = op[3]
        return (((imm >> 5) & 0x7F) << 25) | (FREG[op[1]] << 20) | (
            REG[op[2]] << 15) | (funct3 << 12) | ((imm & 0x1F) << 7) | STORE
    if k == "sw":
        imm = op[3]
        return (((imm >> 5) & 0x7F) << 25) | (REG[op[1]] << 20) | (
            REG[op[2]] << 15) | (0b010 << 12) | ((imm & 0x1F) << 7) | 0x23
    if k == "ldu":
        # QEMU: LOAD funct3 111. The unfrozen chapter puts LDU on LOAD
        # and does not draw the funct3.
        return itype(op[3], op[2], 0b111, op[1], opcode=0x03)
    if k == "lq":
        # QEMU: MISC-MEM funct3 010, rd != 0. The chapter puts LQ on MISC-MEM.
        return itype(op[3], op[2], 0b010, op[1], opcode=0x0F)
    if k == "sq":
        # QEMU: STORE funct3 100. Assembly is sq data, imm(base).
        imm = op[3]
        return (((imm >> 5) & 0x7F) << 25) | (REG[op[1]] << 20) | (
            REG[op[2]] << 15) | (0b100 << 12) | ((imm & 0x1F) << 7) | 0x23
    if k == "addid":
        # QEMU major opcode 1011011. The chapter text names OP-IMM-64.
        return itype(op[3], op[2], 0, op[1], opcode=0x5B)
    if k in ("fsqrt.s", "fsqrt.d"):
        funct7 = 0b0101100 if k == "fsqrt.s" else 0b0101101
        return (funct7 << 25) | (FREG[op[2]] << 15) | (7 << 12) | (
            FREG[op[1]] << 7) | 0x53
    if k == "add":
        return rtype(0, op[3], op[2], 0, op[1])
    if k == "sub":
        return rtype(0x20, op[3], op[2], 0, op[1])
    if k == "addi":
        return itype(op[3], op[2], 0, op[1])
    if k == "slli":
        return itype(op[3], op[2], 0b001, op[1])
    if k == "beq":
        return btype(0, op[1], op[2], labels[op[3]] - pc)
    if k == "bne":
        return btype(1, op[1], op[2], labels[op[3]] - pc)
    if k == "ret":
        return (REG["ra"] << 15) | 0x67
    raise ValueError(op)


def assemble(ops):
    pc = 0
    labels = {}
    body = []
    for op in ops:
        if op[0] == "label":
            labels[op[1]] = pc
        else:
            body.append((pc, op))
            pc += 4
    return [encode(op, at, labels) for at, op in body]


def le_bytes(words):
    return b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in words)


def frame(words, kind):
    """Insert the host escape before every RV parcel."""
    out = bytearray()
    for w in words:
        w &= 0xFFFFFFFF
        hi, lo = (w >> 16) & 0xFFFF, w & 0xFFFF
        if kind == "r800":
            out += bytes((0xED, 0x00)) + struct.pack("<I", w)
        elif kind == "m68k":
            out += struct.pack(">HHH", 0xF400, hi, lo)
        elif kind == "sh2":
            out += struct.pack(">HHH", 0x0000, hi, lo)
        elif kind == "sh4":
            out += struct.pack("<HHH", 0x0000, lo, hi)
        elif kind == "mips-le":
            out += struct.pack("<II", 0x4C000000, w)
        elif kind == "mips-be":
            out += struct.pack(">II", 0x4C000000, w)
        else:
            raise ValueError(kind)
    return bytes(out)


KERNELS = {
    "fill8": {
        "folder": "bx8", "sew": 8, "vlen": 32, "xlen": 32,
        "hosts": ("r800",),
        "ops": [
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 8),
            ("vmv.vx", "v1", "a1"),
            ("vse", 8, "v1", "a2"),
            ("add", "a2", "a2", "t0"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    },
    "copy8": {
        "folder": "bx8", "sew": 8, "vlen": 32, "xlen": 32,
        "hosts": ("r800",),
        "ops": [
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 8),
            ("vle", 8, "v1", "a1"),
            ("vse", 8, "v1", "a2"),
            ("add", "a1", "a1", "t0"),
            ("add", "a2", "a2", "t0"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    },
    "mask8": {
        "folder": "bx8", "sew": 8, "vlen": 32, "xlen": 32,
        "hosts": ("r800",),
        "ops": [
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 8),
            ("vle", 8, "v1", "a2"),
            ("vmsltu.vx", "v0", "v1", "a3"),
            ("vmv.vx", "v2", "a1"),
            ("vse", 8, "v2", "a2", "m"),
            ("add", "a2", "a2", "t0"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    },
    "add8": {
        "folder": "bx8", "sew": 8, "vlen": 32, "xlen": 32,
        "hosts": ("r800",),
        "ops": [
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 8),
            ("vle", 8, "v1", "a1"),
            ("vle", 8, "v2", "a2"),
            ("vsadd.vv", "v3", "v1", "v2"),
            ("vse", 8, "v3", "a3"),
            ("add", "a1", "a1", "t0"),
            ("add", "a2", "a2", "t0"),
            ("add", "a3", "a3", "t0"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    },
    "sew16": {
        "folder": "bx16", "sew": 16, "vlen": 128, "xlen": 32,
        "hosts": ("m68k", "sh2", "sh4"),
        "ops": [
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 16),
            ("vmv.vx", "v1", "a1"),
            ("vse", 16, "v1", "a2"),
            ("slli", "t1", "t0", 1),
            ("add", "a2", "a2", "t1"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    },
    "vwmul16": {
        "folder": "bx16", "sew": 16, "vlen": 128, "xlen": 32,
        "hosts": ("m68k", "sh2", "sh4"),
        "ops": [
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 16),
            ("vle", 16, "v2", "a1"),
            ("vle", 16, "v4", "a2"),
            ("vwmul.vv", "v8", "v2", "v4"),
            ("vsetvli", "t0", "a0", 16),
            ("vnclip.wi", "v1", "v8", 0),
            ("vse", 16, "v1", "a3"),
            ("slli", "t1", "t0", 1),
            ("add", "a1", "a1", "t1"),
            ("add", "a2", "a2", "t1"),
            ("add", "a3", "a3", "t1"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    },
    "stride16": {
        "folder": "bx16", "sew": 16, "vlen": 128, "xlen": 32,
        "hosts": ("m68k", "sh2", "sh4"),
        "ops": [
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 16),
            ("vlse", 16, "v1", "a1", "a2"),
            ("vse", 16, "v1", "a3"),
            ("slli", "t1", "t0", 1),
            ("add", "a3", "a3", "t1"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    },
    "redsum16": {
        "folder": "bx16", "sew": 16, "vlen": 128, "xlen": 32,
        "hosts": ("m68k", "sh2", "sh4"),
        "ops": [
            ("addi", "a2", "zero", 0),
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 16),
            ("vle", 16, "v1", "a1"),
            ("vmv.vi", "v2", 0),
            ("vredsum.vs", "v2", "v1", "v2"),
            ("vmv.xs", "t1", "v2"),
            ("add", "a2", "a2", "t1"),
            ("slli", "t1", "t0", 1),
            ("add", "a1", "a1", "t1"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    },
    "sew32": {
        "folder": "bx32", "sew": 32, "vlen": 128, "xlen": 32,
        "hosts": ("r3000", "vr4300", "ee"),
        "ops": [
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 32),
            ("vmv.vx", "v1", "a1"),
            ("vse", 32, "v1", "a2"),
            ("slli", "t1", "t0", 2),
            ("add", "a2", "a2", "t1"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    },
    "fma32": {
        "folder": "bx32", "sew": 32, "vlen": 128, "xlen": 32,
        "hosts": ("r3000", "vr4300", "ee"),
        "ops": [
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 32),
            ("vle", 32, "v1", "a1"),
            ("vle", 32, "v2", "a2"),
            ("vle", 32, "v3", "a3"),
            ("vfmacc.vv", "v3", "v1", "v2"),
            ("vse", 32, "v3", "a3"),
            ("slli", "t1", "t0", 2),
            ("add", "a1", "a1", "t1"),
            ("add", "a2", "a2", "t1"),
            ("add", "a3", "a3", "t1"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    },
    "gemm32": {
        "folder": "bx32", "sew": 32, "vlen": 128, "xlen": 32,
        "hosts": ("r3000", "vr4300", "ee"),
        "ops": [
            ("beq", "a0", "zero", "end"),
            ("vsetvli", "zero", "zero", 32),
            ("vle", 32, "v8", "a3"),
            ("addi", "t0", "a3", 16),
            ("vle", 32, "v9", "t0"),
            ("addi", "t0", "a3", 32),
            ("vle", 32, "v10", "t0"),
            ("addi", "t0", "a3", 48),
            ("vle", 32, "v11", "t0"),
            ("label", "loop"),
            ("vle", 32, "v1", "a1"),
            ("flw", "ft0", "a2", 0),
            ("vfmacc.vf", "v8", "ft0", "v1"),
            ("flw", "ft0", "a2", 4),
            ("vfmacc.vf", "v9", "ft0", "v1"),
            ("flw", "ft0", "a2", 8),
            ("vfmacc.vf", "v10", "ft0", "v1"),
            ("flw", "ft0", "a2", 12),
            ("vfmacc.vf", "v11", "ft0", "v1"),
            ("addi", "a1", "a1", 16),
            ("addi", "a2", "a2", 16),
            ("addi", "a0", "a0", -1),
            ("bne", "a0", "zero", "loop"),
            ("vse", 32, "v8", "a3"),
            ("addi", "t0", "a3", 16),
            ("vse", 32, "v9", "t0"),
            ("addi", "t0", "a3", 32),
            ("vse", 32, "v10", "t0"),
            ("addi", "t0", "a3", 48),
            ("vse", 32, "v11", "t0"),
            ("label", "end"),
            ("ret",),
        ],
    },
    "index32": {
        "folder": "bx32", "sew": 32, "vlen": 128, "xlen": 32,
        "hosts": ("r3000", "vr4300", "ee"),
        "ops": [
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 32),
            ("vle", 32, "v2", "a2"),
            ("vluxei", 32, "v1", "a1", "v2"),
            ("vse", 32, "v1", "a3"),
            ("slli", "t1", "t0", 2),
            ("add", "a2", "a2", "t1"),
            ("add", "a3", "a3", "t1"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    },
    "sew64": {
        "folder": "bx64", "sew": 64, "vlen": 128, "xlen": 64,
        "hosts": (),
        "ops": [
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 64),
            ("vmv.vx", "v1", "a1"),
            ("vse", 64, "v1", "a2"),
            ("slli", "t1", "t0", 3),
            ("add", "a2", "a2", "t1"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    },
    "fma64": {
        "folder": "bx64", "sew": 64, "vlen": 128, "xlen": 64,
        "hosts": (),
        "ops": [
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 64),
            ("vle", 64, "v1", "a1"),
            ("vle", 64, "v2", "a2"),
            ("vle", 64, "v3", "a3"),
            ("vfmacc.vv", "v3", "v1", "v2"),
            ("vse", 64, "v3", "a3"),
            ("slli", "t1", "t0", 3),
            ("add", "a1", "a1", "t1"),
            ("add", "a2", "a2", "t1"),
            ("add", "a3", "a3", "t1"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    },
    "gemm64": {
        "folder": "bx64", "sew": 64, "vlen": 128, "xlen": 64,
        "hosts": (),
        "ops": [
            ("beq", "a0", "zero", "end"),
            ("vsetvli", "zero", "zero", 64),
            ("vle", 64, "v8", "a3"),
            ("addi", "t0", "a3", 16),
            ("vle", 64, "v9", "t0"),
            ("addi", "t0", "a3", 32),
            ("vle", 64, "v10", "t0"),
            ("addi", "t0", "a3", 48),
            ("vle", 64, "v11", "t0"),
            ("label", "loop"),
            ("vle", 64, "v1", "a1"),
            ("fld", "ft0", "a2", 0),
            ("vfmacc.vf", "v8", "ft0", "v1"),
            ("fld", "ft0", "a2", 8),
            ("vfmacc.vf", "v9", "ft0", "v1"),
            ("fld", "ft0", "a2", 16),
            ("vfmacc.vf", "v10", "ft0", "v1"),
            ("fld", "ft0", "a2", 24),
            ("vfmacc.vf", "v11", "ft0", "v1"),
            ("addi", "a1", "a1", 16),
            ("addi", "a2", "a2", 32),
            ("addi", "a0", "a0", -1),
            ("bne", "a0", "zero", "loop"),
            ("vse", 64, "v8", "a3"),
            ("addi", "t0", "a3", 16),
            ("vse", 64, "v9", "t0"),
            ("addi", "t0", "a3", 32),
            ("vse", 64, "v10", "t0"),
            ("addi", "t0", "a3", 48),
            ("vse", 64, "v11", "t0"),
            ("label", "end"),
            ("ret",),
        ],
    },
    "index64": {
        "folder": "bx64", "sew": 64, "vlen": 128, "xlen": 64,
        "hosts": (),
        "ops": [
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 64),
            ("vle", 64, "v2", "a2"),
            ("vluxei", 64, "v1", "a1", "v2"),
            ("vse", 64, "v1", "a3"),
            ("slli", "t1", "t0", 3),
            ("add", "a2", "a2", "t1"),
            ("add", "a3", "a3", "t1"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    },
}

HOST_KIND = {
    "r800": "r800",
    "m68k": "m68k",
    "sh2": "sh2",
    "sh4": "sh4",
    "r3000": "mips-le",
    "ee": "mips-le",
    "vr4300": "mips-be",
}

HOST_END = {
    "r800": "le", "m68k": "be", "sh2": "be", "sh4": "le",
    "r3000": "le", "ee": "le", "vr4300": "be",
}

README = {
    "bx8": (
        "RV32 hart image, SVM-8. Little-endian 32-bit parcels. "
        "SEW 8, LMUL 1, tail undisturbed, mask undisturbed. "
        "VLEN 32 and XLEN 32 are the profile, not fields in the bytes. "
        "Host: none in this folder."
    ),
    "bx8/r800": (
        "Harness image for an R800 softcore. Each RV parcel, scalar or vector, "
        "is preceded by the escape ED 00. Bytes are little-endian. "
        "A ROM escapes only OP-V and writes the branch and the address "
        "update in R800 instructions. The rs1 field here is still the RV "
        "number. Re-assemble it as the host index before you put it in a ROM. "
        "Map: 0 BC, 1 DE, 2 HL, 3 IX, 4 IY, 5 SP."
    ),
    "bx16": (
        "RV32 hart image, SVM-16. Little-endian 32-bit parcels. "
        "SEW is the filename: 16 on the halfword images, 8 on sw8. "
        "LMUL 1, tail undisturbed, mask undisturbed. "
        "VLEN 128, XLEN 32. Host: none in this folder."
    ),
    "bx16/m68k": (
        "Harness image for a 68000 softcore. Each RV parcel is preceded by "
        "the F-line word F400 (CpID 2). The parcel follows as two big-endian "
        "halfwords, high half first. A ROM escapes only OP-V. "
        "rs1[3:0] selects D0-D7 then A0-A7. This file still holds RV numbers."
    ),
    "bx16/sh2": (
        "Harness image for an SH-2 softcore, Saturn endianness. Escape halfword "
        "0000, then the parcel as two big-endian halfwords, high half first. "
        "SH-2 is integer only on SVM-16. Do not use this image as a Dreamcast "
        "stream: the Dreamcast is little-endian and lives in sh4/."
    ),
    "bx16/sh4": (
        "Harness image for an SH-4 softcore, Dreamcast endianness. Escape "
        "halfword 0000, then the parcel as two little-endian halfwords, low "
        "half first. Integer injection only. The Fxxx FPU space is not the escape."
    ),
    "bx32": (
        "RV32 hart image, Zve32f. Little-endian 32-bit parcels. "
        "Tail undisturbed, mask undisturbed. VLEN 128 is the acceptance floor. "
        "XLEN 32. Host: none in this folder."
    ),
    "bx32/r3000": (
        "Harness image for a PlayStation R3000A. Escape word 4C000000 (COP3, "
        "all other fields zero), little-endian, then the parcel little-endian. "
        "The adaptor claims the word before Coprocessor Unusable runs. "
        "COP2 is the GTE and is left alone."
    ),
    "bx32/vr4300": (
        "Harness image for a Nintendo 64 VR4300. Same COP3 escape 4C000000, "
        "big-endian, parcel big-endian. COP2 is unused on this CPU and is "
        "still not the escape, so the three MIPS hosts share one opcode."
    ),
    "bx32/ee": (
        "Harness image for a PlayStation 2 Emotion Engine MIPS core. COP3 "
        "escape 4C000000, little-endian. VU0 is COP2 and is left alone. "
        "The adaptor reads the low 32 bits of the GPR as the address when "
        "the attached SVM is BX-32."
    ),
    "bx64": (
        "RV64 hart image, V or Zve64d. Little-endian 32-bit instruction "
        "parcels (the instruction word is still 32 bits). SEW as named, "
        "LMUL 1, tail undisturbed, mask undisturbed. VLEN 128 is the floor. "
        "XLEN 64. No host frame in this edition; the BX-32 COP3 framing is "
        "the pattern, with the low 64 bits of a 64-bit GPR as the address."
    ),
    "bx512": (
        "RV64 hart image, V or Zve64d, VLEN pinned to 512. ELEN stays 64. "
        "Little-endian 32-bit instruction parcels. The panel stride in these "
        "kernels is vl, so the same words are legal at a smaller VLEN. "
        "The filename records the register length this image was accepted on."
    ),
    "rv128": (
        "RV128I hart image. Little-endian 32-bit instruction parcels. "
        "XLEN 128. No vector parcel, so the filename records vlen0. "
        "sew64 is that same filename slot: ELEN stays 64 if a vector unit "
        "is attached later, and SEW 128 is not encoded. "
        "lq, sq, ldu, and addid follow QEMU's insn32.decode. "
        "addi is the XLEN add. slli by 64 uses the low 7 bits of the "
        "I-immediate. No host frame. These files are not ROMs."
    ),
}


def _mine(sew, inner, ptrs, prefix=(), tail=()):
    shift = {32: 2, 64: 3}[sew]
    ops = list(prefix)
    ops += [
        ("beq", "a0", "zero", "end"),
        ("label", "loop"),
        ("vsetvli", "t0", "a0", sew),
    ]
    ops += list(inner)
    ops.append(("slli", "t1", "t0", shift))
    for ptr in ptrs:
        ops.append(("add", ptr, ptr, "t1"))
    ops += [
        ("sub", "a0", "a0", "t0"),
        ("bne", "a0", "zero", "loop"),
        ("label", "end"),
    ]
    ops += list(tail)
    ops.append(("ret",))
    return ops


def _blas_kernels():
    """Level-1 and level-2 bodies. s* is SEW 32, d* is SEW 64."""
    for sew, folder, xlen, hosts in (
        (32, "bx32", 32, ("r3000", "vr4300", "ee")),
        (64, "bx64", 64, ()),
    ):
        ld = "flw" if sew == 32 else "fld"
        st = "fsw" if sew == 32 else "fsd"
        sq = "fsqrt.s" if sew == 32 else "fsqrt.d"
        tag = str(sew)

        def put(name, comment, ops, _sew=sew, _folder=folder, _xlen=xlen, _hosts=hosts):
            KERNELS[name] = {
                "folder": _folder, "sew": _sew, "vlen": 128, "xlen": _xlen,
                "hosts": _hosts, "ops": ops,
            }
            HEAD[name] = comment

        put(f"scal{tag}", "a0 = n, a1 = x, a2 = alpha. x *= alpha", _mine(
            sew,
            [("vle", sew, "v1", "a1"),
             ("vfmul.vf", "v1", "v1", "ft0"),
             ("vse", sew, "v1", "a1")],
            ("a1",),
            prefix=[(ld, "ft0", "a2", 0)],
        ))
        put(f"copy{tag}", "a0 = n, a1 = src, a2 = dst", _mine(
            sew,
            [("vle", sew, "v1", "a1"), ("vse", sew, "v1", "a2")],
            ("a1", "a2"),
        ))
        put(f"axpy{tag}", "a0 = n, a1 = x, a2 = y, a3 = alpha. y += alpha*x", _mine(
            sew,
            [("vle", sew, "v1", "a1"),
             ("vle", sew, "v2", "a2"),
             ("vfmacc.vf", "v2", "ft0", "v1"),
             ("vse", sew, "v2", "a2")],
            ("a1", "a2"),
            prefix=[(ld, "ft0", "a3", 0)],
        ))
        put(f"dot{tag}", "a0 = n, a1 = x, a2 = y, a3 = sum out", _mine(
            sew,
            [("vle", sew, "v1", "a1"),
             ("vle", sew, "v2", "a2"),
             ("vfmul.vv", "v1", "v1", "v2"),
             ("vfredusum.vs", "v4", "v1", "v4")],
            ("a1", "a2"),
            prefix=[("vmv.vi", "v4", 0)],
            tail=[("vfmv.fs", "ft0", "v4"), (st, "ft0", "a3", 0)],
        ))
        put(f"nrm2{tag}", "a0 = n, a1 = x, a2 = norm out. sqrt of sum of squares", _mine(
            sew,
            [("vle", sew, "v1", "a1"),
             ("vfmul.vv", "v1", "v1", "v1"),
             ("vfredusum.vs", "v4", "v1", "v4")],
            ("a1",),
            prefix=[("vmv.vi", "v4", 0)],
            tail=[("vfmv.fs", "ft0", "v4"), (sq, "ft0", "ft0"), (st, "ft0", "a2", 0)],
        ))
        put(f"asum{tag}", "a0 = n, a1 = x, a2 = sum of absolute values", _mine(
            sew,
            [("vle", sew, "v1", "a1"),
             ("vfsgnjx.vv", "v1", "v1", "v1"),
             ("vfredusum.vs", "v4", "v1", "v4")],
            ("a1",),
            prefix=[("vmv.vi", "v4", 0)],
            tail=[("vfmv.fs", "ft0", "v4"), (st, "ft0", "a2", 0)],
        ))
        put(f"swap{tag}", "a0 = n, a1 = x, a2 = y", _mine(
            sew,
            [("vle", sew, "v1", "a1"),
             ("vle", sew, "v2", "a2"),
             ("vse", sew, "v2", "a1"),
             ("vse", sew, "v1", "a2")],
            ("a1", "a2"),
        ))
        put(f"rot{tag}", "a0 = n, a1 = x, a2 = y, a3 = (c, s)", _mine(
            sew,
            [("vle", sew, "v1", "a1"),
             ("vle", sew, "v2", "a2"),
             ("vfmul.vf", "v3", "v1", "ft0"),
             ("vfmul.vf", "v4", "v2", "ft1"),
             ("vfadd.vv", "v5", "v3", "v4"),
             ("vfmul.vf", "v3", "v2", "ft0"),
             ("vfmul.vf", "v4", "v1", "ft1"),
             ("vfsub.vv", "v6", "v3", "v4"),
             ("vse", sew, "v5", "a1"),
             ("vse", sew, "v6", "a2")],
            ("a1", "a2"),
            prefix=[(ld, "ft0", "a3", 0), (ld, "ft1", "a3", 4 if sew == 32 else 8)],
        ))
        # One register tall. n is the count of columns, not of rows.
        put(f"gemv{tag}", "a0 = columns, a1 = A, a2 = x, a3 = y. rows = VLMAX, lda = VLMAX", [
            ("beq", "a0", "zero", "end"),
            ("vsetvli", "t0", "zero", sew),
            ("slli", "t1", "t0", 2 if sew == 32 else 3),
            ("vle", sew, "v8", "a3"),
            ("label", "loop"),
            ("vle", sew, "v1", "a1"),
            (ld, "ft0", "a2", 0),
            ("vfmacc.vf", "v8", "ft0", "v1"),
            ("add", "a1", "a1", "t1"),
            ("addi", "a2", "a2", 4 if sew == 32 else 8),
            ("addi", "a0", "a0", -1),
            ("bne", "a0", "zero", "loop"),
            ("vse", sew, "v8", "a3"),
            ("label", "end"),
            ("ret",),
        ])
        put(f"ger{tag}", "a0 = columns, a1 = x, a2 = y, a3 = A. A += x y^T, lda = VLMAX", [
            ("beq", "a0", "zero", "end"),
            ("vsetvli", "t0", "zero", sew),
            ("slli", "t1", "t0", 2 if sew == 32 else 3),
            ("vle", sew, "v1", "a1"),
            ("label", "loop"),
            ("vle", sew, "v8", "a3"),
            (ld, "ft0", "a2", 0),
            ("vfmacc.vf", "v8", "ft0", "v1"),
            ("vse", sew, "v8", "a3"),
            ("add", "a3", "a3", "t1"),
            ("addi", "a2", "a2", 4 if sew == 32 else 8),
            ("addi", "a0", "a0", -1),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ])
        put(f"iamax{tag}", "a0 = n (<= VLMAX), a1 = x, a2 = index out. First index of max |x|", [
            ("beq", "a0", "zero", "end"),
            ("vsetvli", "t0", "a0", sew),
            ("vle", sew, "v1", "a1"),
            ("vfsgnjx.vv", "v1", "v1", "v1"),
            ("vfredmax.vs", "v2", "v1", "v1"),
            ("vfmv.fs", "ft0", "v2"),
            ("vmfeq.vf", "v3", "v1", "ft0"),
            ("vfirst", "t0", "v3"),
            ("sw", "t0", "a2", 0),
            ("label", "end"),
            ("ret",),
        ])


def _wide(sew):
    ld = "flw" if sew == 32 else "fld"
    bstep = 4 if sew == 32 else 8
    shift = 2 if sew == 32 else 3
    return [
        ("beq", "a0", "zero", "end"),
        ("vsetvli", "t0", "zero", sew),
        ("slli", "t1", "t0", shift),
        ("vle", sew, "v8", "a3"),
        ("add", "t0", "a3", "t1"),
        ("vle", sew, "v9", "t0"),
        ("add", "t0", "t0", "t1"),
        ("vle", sew, "v10", "t0"),
        ("add", "t0", "t0", "t1"),
        ("vle", sew, "v11", "t0"),
        ("label", "loop"),
        ("vle", sew, "v1", "a1"),
        (ld, "ft0", "a2", 0),
        ("vfmacc.vf", "v8", "ft0", "v1"),
        (ld, "ft0", "a2", bstep),
        ("vfmacc.vf", "v9", "ft0", "v1"),
        (ld, "ft0", "a2", 2 * bstep),
        ("vfmacc.vf", "v10", "ft0", "v1"),
        (ld, "ft0", "a2", 3 * bstep),
        ("vfmacc.vf", "v11", "ft0", "v1"),
        ("add", "a1", "a1", "t1"),
        ("addi", "a2", "a2", 4 * bstep),
        ("addi", "a0", "a0", -1),
        ("bne", "a0", "zero", "loop"),
        ("vse", sew, "v8", "a3"),
        ("add", "t0", "a3", "t1"),
        ("vse", sew, "v9", "t0"),
        ("add", "t0", "t0", "t1"),
        ("vse", sew, "v10", "t0"),
        ("add", "t0", "t0", "t1"),
        ("vse", sew, "v11", "t0"),
        ("label", "end"),
        ("ret",),
    ]


def _app_kernels():
    """Loops that consumer and scientific code call by another name."""

    def put(name, folder, sew, vlen, xlen, hosts, comment, ops):
        KERNELS[name] = {
            "folder": folder, "sew": sew, "vlen": vlen, "xlen": xlen,
            "hosts": hosts, "ops": ops,
        }
        HEAD[name] = comment

    put(
        "sad16", "bx16", 16, 128, 32, ("m68k", "sh2", "sh4"),
        "a0 = n, a1 = src0, a2 = src1, a3 = sum. Sum of |a-b|. Each strip wraps in 16 bits.",
        [
            ("addi", "a3", "zero", 0),
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 16),
            ("vle", 16, "v1", "a1"),
            ("vle", 16, "v2", "a2"),
            ("vsub.vv", "v3", "v1", "v2"),
            ("vsub.vv", "v4", "v2", "v1"),
            ("vmax.vv", "v3", "v3", "v4"),
            ("vmv.vi", "v4", 0),
            ("vredsum.vs", "v4", "v3", "v4"),
            ("vmv.xs", "t1", "v4"),
            ("add", "a3", "a3", "t1"),
            ("slli", "t1", "t0", 1),
            ("add", "a1", "a1", "t1"),
            ("add", "a2", "a2", "t1"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    )
    put(
        "sw8", "bx16", 8, 128, 32, ("m68k", "sh2", "sh4"),
        "a0 = n, a1 = H, a2 = profile, a3 = E. Saturating add, then unsigned max.",
        [
            ("beq", "a0", "zero", "end"),
            ("label", "loop"),
            ("vsetvli", "t0", "a0", 8),
            ("vle", 8, "v1", "a1"),
            ("vle", 8, "v2", "a2"),
            ("vsaddu.vv", "v1", "v1", "v2"),
            ("vle", 8, "v3", "a3"),
            ("vmaxu.vv", "v1", "v1", "v3"),
            ("vse", 8, "v1", "a1"),
            ("add", "a1", "a1", "t0"),
            ("add", "a2", "a2", "t0"),
            ("add", "a3", "a3", "t0"),
            ("sub", "a0", "a0", "t0"),
            ("bne", "a0", "zero", "loop"),
            ("label", "end"),
            ("ret",),
        ],
    )
    put(
        "wmul32", "bx32", 32, 128, 32, ("r3000", "vr4300", "ee"),
        "a0 = n, a1 = x, a2 = window, a3 = dst. Element-wise product.",
        _mine(32, [
            ("vle", 32, "v1", "a1"),
            ("vle", 32, "v2", "a2"),
            ("vfmul.vv", "v1", "v1", "v2"),
            ("vse", 32, "v1", "a3"),
        ], ("a1", "a2", "a3")),
    )
    put(
        "bfly32", "bx32", 32, 128, 32, ("r3000", "vr4300", "ee"),
        "a0 = n, a1 = a, a2 = b, a3 = twiddle. Real radix-2 butterfly, in place.",
        _mine(32, [
            ("vle", 32, "v1", "a1"),
            ("vle", 32, "v2", "a2"),
            ("vfmul.vf", "v3", "v2", "ft0"),
            ("vfadd.vv", "v4", "v1", "v3"),
            ("vfsub.vv", "v5", "v1", "v3"),
            ("vse", 32, "v4", "a1"),
            ("vse", 32, "v5", "a2"),
        ], ("a1", "a2"), prefix=[("flw", "ft0", "a3", 0)]),
    )
    put(
        "cmul32", "bx32", 32, 128, 32, ("r3000", "vr4300", "ee"),
        "a0 = n, a1 = re, a2 = im, a3 = (wr, wi). Split complex times one scalar.",
        _mine(32, [
            ("vle", 32, "v1", "a1"),
            ("vle", 32, "v2", "a2"),
            ("vfmul.vf", "v3", "v1", "ft0"),
            ("vfmul.vf", "v4", "v2", "ft1"),
            ("vfsub.vv", "v5", "v3", "v4"),
            ("vfmul.vf", "v3", "v1", "ft1"),
            ("vfmul.vf", "v4", "v2", "ft0"),
            ("vfadd.vv", "v6", "v3", "v4"),
            ("vse", 32, "v5", "a1"),
            ("vse", 32, "v6", "a2"),
        ], ("a1", "a2"), prefix=[
            ("flw", "ft0", "a3", 0),
            ("flw", "ft1", "a3", 4),
        ]),
    )


HEAD = {
    "fill8": "a0 = n, a1 = fill byte, a2 = dst",
    "copy8": "a0 = n, a1 = src, a2 = dst",
    "mask8": "a0 = n, a1 = fill, a2 = dst, a3 = threshold",
    "add8": "a0 = n, a1 = src0, a2 = src1, a3 = dst",
    "sew16": "a0 = n (elements), a1 = fill halfword, a2 = dst",
    "vwmul16": "a0 = n, a1 = src0, a2 = src1, a3 = dst. Wide product, then clip.",
    "stride16": "a0 = n, a1 = base, a2 = stride (bytes), a3 = dst. Base stays.",
    "redsum16": "a0 = n, a1 = src, a2 = sum out. Each strip wraps in 16 bits.",
    "sew32": "a0 = n, a1 = fill word, a2 = dst",
    "fma32": "a0 = n, a1 = a, a2 = b, a3 = c.  c = c + a*b",
    "gemm32": "a0 = K, a1 = A panel, a2 = B panel, a3 = C. 4-wide outer product.",
    "index32": "a0 = n, a1 = base, a2 = index vector, a3 = dst",
    "sew64": "a0 = n, a1 = fill, a2 = dst",
    "fma64": "a0 = n, a1 = a, a2 = b, a3 = c.  c = c + a*b",
    "gemm64": "a0 = K, a1 = A panel, a2 = B panel, a3 = C. 2-wide outer product.",
    "index64": "a0 = n, a1 = base, a2 = index vector, a3 = dst",
}

_blas_kernels()
_app_kernels()

for _sew, _comment in (
    (32, "a0 = K, a1 = A, a2 = B, a3 = C. 4 columns, stride = vl."),
    (64, "a0 = K, a1 = A, a2 = B, a3 = C. 4 columns, stride = vl."),
):
    _name = f"wide{_sew}"
    KERNELS[_name] = {
        "folder": "bx512", "sew": _sew, "vlen": 512, "xlen": 64,
        "hosts": (), "ops": _wide(_sew),
    }
    HEAD[_name] = _comment


def _rv128_kernels():
    """Integer kernels for the unfrozen RV128I base. No vector parcel."""
    specs = (
        ("quad", "a0 = src, a1 = dst. One quadword, then both pointers add 16.", [
            ("lq", "t0", "a0", 0),
            ("sq", "t0", "a1", 0),
            ("addi", "a0", "a0", 16),
            ("addi", "a1", "a1", 16),
            ("ret",),
        ]),
        ("add128", "a0 = src, a1 = dst. Quadword plus 16. Carry enters bit 64.", [
            ("lq", "t0", "a0", 0),
            ("addi", "t0", "t0", 16),
            ("sq", "t0", "a1", 0),
            ("ret",),
        ]),
        ("addd16", "a0 = src, a1 = dst. Low 64 bits plus 16, then sign-extend.", [
            ("lq", "t0", "a0", 0),
            ("addid", "t0", "t0", 16),
            ("sq", "t0", "a1", 0),
            ("ret",),
        ]),
        ("zext64", "a0 = doubleword, a1 = quadword. High 64 bits of the store are 0.", [
            ("ldu", "t0", "a0", 0),
            ("sq", "t0", "a1", 0),
            ("ret",),
        ]),
        ("sh64", "a0 = src, a1 = dst. Quadword shifted left by 64.", [
            ("lq", "t0", "a0", 0),
            ("slli", "t0", "t0", 64),
            ("sq", "t0", "a1", 0),
            ("ret",),
        ]),
    )
    for name, comment, ops in specs:
        KERNELS[name] = {
            "folder": "rv128", "sew": 64, "vlen": 0, "xlen": 128,
            "hosts": (), "ops": ops,
        }
        HEAD[name] = comment


_rv128_kernels()


def asm_parts(op):
    """Mnemonic and operands, in the form the chapters print."""
    k = op[0]
    if k == "vsetvli":
        return "vsetvli", f"{op[1]}, {op[2]}, e{op[3]}, m1, tu, mu"
    if k == "vmv.vx":
        return "vmv.v.x", f"{op[1]}, {op[2]}"
    if k == "vmv.vi":
        return "vmv.v.i", f"{op[1]}, {op[2]}"
    if k == "vmv.xs":
        return "vmv.x.s", f"{op[1]}, {op[2]}"
    if k == "vle":
        return f"vle{op[1]}.v", f"{op[2]}, ({op[3]})"
    if k == "vse":
        mask = ", v0.t" if len(op) > 4 else ""
        return f"vse{op[1]}.v", f"{op[2]}, ({op[3]}){mask}"
    if k == "vlse":
        return f"vlse{op[1]}.v", f"{op[2]}, ({op[3]}), {op[4]}"
    if k == "vluxei":
        return f"vluxei{op[1]}.v", f"{op[2]}, ({op[3]}), {op[4]}"
    if k in ("vsadd.vv", "vsaddu.vv", "vsub.vv", "vmax.vv", "vmaxu.vv",
             "vwmul.vv", "vredsum.vs", "vfmacc.vv", "vfmacc.vf",
             "vfmul.vv", "vfmul.vf", "vfadd.vv", "vfsub.vv", "vfsgnjx.vv",
             "vfredusum.vs", "vfredmax.vs", "vmfeq.vf"):
        return k, f"{op[1]}, {op[2]}, {op[3]}"
    if k == "vfmv.fs":
        return "vfmv.f.s", f"{op[1]}, {op[2]}"
    if k == "vfirst":
        return "vfirst.m", f"{op[1]}, {op[2]}"
    if k in ("flw", "fld", "fsw", "fsd", "sw", "lq", "sq", "ldu"):
        return k, f"{op[1]}, {op[3]}({op[2]})"
    if k in ("fsqrt.s", "fsqrt.d"):
        return k, f"{op[1]}, {op[2]}"
    if k == "vmsltu.vx":
        return "vmsltu.vx", f"{op[1]}, {op[2]}, {op[3]}"
    if k == "vnclip.wi":
        return "vnclip.wi", f"{op[1]}, {op[2]}, {op[3]}"
    if k == "add":
        return "add", f"{op[1]}, {op[2]}, {op[3]}"
    if k == "sub":
        return "sub", f"{op[1]}, {op[2]}, {op[3]}"
    if k == "addi" and op[2] == "zero" and op[3] == 0:
        return "li", f"{op[1]}, 0"
    if k in ("addi", "addid"):
        return k, f"{op[1]}, {op[2]}, {op[3]}"
    if k == "slli":
        return "slli", f"{op[1]}, {op[2]}, {op[3]}"
    if k == "beq" and op[2] == "zero":
        return "beqz", f"{op[1]}, {op[3]}"
    if k == "bne" and op[2] == "zero":
        return "bnez", f"{op[1]}, {op[3]}"
    if k == "beq":
        return "beq", f"{op[1]}, {op[2]}, {op[3]}"
    if k == "bne":
        return "bne", f"{op[1]}, {op[2]}, {op[3]}"
    if k == "ret":
        return "ret", ""
    raise ValueError(op)


def listing_row(addr, word, text):
    cell = f"{word:08X}" if word is not None else "        "
    return f"{addr:04X}: {cell} | {text}"


def listing_lines(name, meta, words):
    gutter = " " * listing_row(0, None, "").index("|")
    lines = [
        f"{gutter}| # {HEAD[name]}",
        f"{gutter}| {name}:",
    ]
    pc = 0
    wi = 0
    for op in meta["ops"]:
        if op[0] == "label":
            lines.append(listing_row(pc, None, f"{op[1]}:"))
        else:
            mnem, args = asm_parts(op)
            text = f"{mnem:<12}{args}".rstrip()
            lines.append(listing_row(pc, words[wi], text))
            wi += 1
            pc += 4
    pipes = [ln.index("|") for ln in lines]
    if len(set(pipes)) != 1:
        raise SystemExit(f"{name}: bar column {pipes}")
    return lines


def fname(name, meta, endian):
    return (
        f"{name}.sew{meta['sew']}.lmul1.tu.mu."
        f"vlen{meta['vlen']}.xlen{meta['xlen']}.{endian}.bin"
    )


def hex_lines(words):
    lines = []
    for i, w in enumerate(words):
        lines.append(f"{i*4:04X}  {w & 0xFFFFFFFF:08X}")
    return lines


def main():
    # Anchors a reader can recompute by hand.
    assert vsetvli("zero", "zero", 8) == 0x00007057
    assert encode(("ret",), 0, {}) == 0x00008067
    assert encode(("addi", "a2", "zero", 0), 0, {}) == 0x00000613
    assert encode(("vfmacc.vf", "v8", "ft0", "v1"), 0, {}) == 0xB2105457
    assert encode(("flw", "ft0", "a2", 0), 0, {}) == 0x00062007
    assert encode(("fld", "ft0", "a1", 0), 0, {}) == 0x0005B007
    assert encode(("vfsgnjx.vv", "v1", "v1", "v1"), 0, {}) == 0x2A1090D7
    assert encode(("vfirst", "t0", "v3"), 0, {}) == 0x4238A2D7
    assert encode(("sw", "t0", "a2", 0), 0, {}) == 0x00562023
    assert encode(("fsqrt.s", "ft0", "ft0"), 0, {}) == 0x58007053
    # RV128I anchors. lq/sq/ldu/addid are the QEMU field placement.
    # slli by 4 matches RV64. slli by 64 sets the seventh shamt bit.
    assert encode(("lq", "t0", "a0", 0), 0, {}) == 0x0005228F
    assert encode(("sq", "t0", "a1", 0), 0, {}) == 0x0055C023
    assert encode(("ldu", "t0", "a0", 0), 0, {}) == 0x00057283
    assert encode(("addi", "a0", "a0", 16), 0, {}) == 0x01050513
    assert encode(("addi", "t0", "t0", 16), 0, {}) == 0x01028293
    assert encode(("addid", "a0", "a0", 16), 0, {}) == 0x0105055B
    assert encode(("addid", "t0", "t0", 16), 0, {}) == 0x010282DB
    assert encode(("slli", "a0", "a0", 4), 0, {}) == 0x00451513
    assert encode(("slli", "a0", "a0", 64), 0, {}) == 0x04051513
    assert encode(("slli", "t0", "t0", 64), 0, {}) == 0x04029293

    gen = ROOT / "gen"
    gen.mkdir(exist_ok=True)
    readme_dirs = {}

    for name, meta in KERNELS.items():
        words = assemble(meta["ops"])
        endian = "le"
        raw_name = fname(name, meta, endian)
        folder = ROOT / meta["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / raw_name).write_bytes(le_bytes(words))
        readme_dirs.setdefault(meta["folder"], []).append(raw_name)

        body = "\n".join(listing_lines(name, meta, words))
        rel = f"kernels/{meta['folder']}/{raw_name}"
        (gen / f"{name}.tex").write_text(
            "% Generated by encode_kernels.py. Do not edit.\n"
            "% verbatim, not listings: the bar column is spaces, and listings drops them.\n"
            "\\begingroup\n"
            "\\makeatletter\n"
            "\\def\\verbatim@font{\\footnotesize\\ttfamily}%\n"
            "\\begin{verbatim}\n"
            f"{body}\n"
            "\\end{verbatim}\n"
            "\\endgroup\n"
            f"\\noindent\\path{{{rel}}}\n",
            encoding="utf-8",
        )

        for host in meta["hosts"]:
            kind = HOST_KIND[host]
            data = frame(words, kind)
            hfolder = folder / host
            hfolder.mkdir(parents=True, exist_ok=True)
            hname = fname(name, meta, HOST_END[host])
            (hfolder / hname).write_bytes(data)
            key = f"{meta['folder']}/{host}"
            readme_dirs.setdefault(key, []).append(hname)

    for key, files in readme_dirs.items():
        text = README[key] + "\n\n" + "\n".join(f"- `{f}`" for f in files) + "\n"
        (ROOT / key / "README.md").write_text(text, encoding="utf-8")

    print(f"wrote {len(KERNELS)} kernels")


if __name__ == "__main__":
    main()
