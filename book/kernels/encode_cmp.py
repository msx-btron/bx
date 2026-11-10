#!/usr/bin/env python3
"""Build the three-ISA comparison tables.

Each row is one instruction: ISA, address, hex, assembly.
RVV hex comes from encode_kernels.py. Intel and AArch64 hex come from clang.
The ARM column is AArch64 NEON because that is the encoding this assembler emits.
SVE is not in these bytes.
"""

import importlib.util
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
AVX512 = {"wide32", "wide64"}
RV128 = {"quad", "add128", "addd16", "zext64", "sh64"}
spec = importlib.util.spec_from_file_location("enc", ROOT / "encode_kernels.py")
enc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enc)

# Inner strips. Addresses start at 0. Intel syntax, destination first.
INTEL = {
    "fill8": [
        "movdqu [rdi], xmm0",
        "add rdi, 16",
        "sub rsi, 16",
        "ja fill8",
    ],
    "copy8": [
        "movdqu xmm0, [rsi]",
        "movdqu [rdi], xmm0",
        "add rsi, 16",
        "add rdi, 16",
        "sub rdx, 16",
        "ja copy8",
    ],
    "mask8": [
        "movdqu xmm0, [rdi]",
        "movdqa xmm1, xmm6",
        "psubusb xmm1, xmm0",
        "pxor xmm2, xmm2",
        "pcmpeqb xmm1, xmm2",
        "pand xmm0, xmm1",
        "pandn xmm1, xmm5",
        "por xmm0, xmm1",
        "movdqu [rdi], xmm0",
    ],
    "add8": [
        "movdqu xmm0, [rsi]",
        "movdqu xmm1, [rdx]",
        "paddusb xmm0, xmm1",
        "movdqu [rdi], xmm0",
    ],
    "sew16": [
        "movd xmm0, esi",
        "pshuflw xmm0, xmm0, 0",
        "punpcklqdq xmm0, xmm0",
        "movdqu [rdi], xmm0",
        "add rdi, 16",
        "sub rdx, 8",
        "ja sew16",
    ],
    "vwmul16": [
        "movdqa xmm2, xmm0",
        "pmullw xmm0, xmm1",
        "pmulhw xmm2, xmm1",
        "movdqa xmm3, xmm0",
        "punpcklwd xmm0, xmm2",
        "punpckhwd xmm3, xmm2",
        "packssdw xmm0, xmm3",
    ],
    "stride16": [
        "pinsrw xmm0, [rsi], 0",
        "add rsi, rdx",
        "pinsrw xmm0, [rsi], 1",
        "add rsi, rdx",
        "pinsrw xmm0, [rsi], 2",
        "add rsi, rdx",
        "pinsrw xmm0, [rsi], 3",
        "movdqu [rdi], xmm0",
    ],
    "redsum16": [
        "phaddw xmm0, xmm0",
        "phaddw xmm0, xmm0",
        "phaddw xmm0, xmm0",
        "pextrw eax, xmm0, 0",
    ],
    "sew32": [
        "movd xmm0, esi",
        "pshufd xmm0, xmm0, 0",
        "movdqu [rdi], xmm0",
        "add rdi, 16",
        "sub rdx, 4",
        "ja sew32",
    ],
    "fma32": [
        "vmovups ymm0, [rsi]",
        "vmovups ymm1, [rdx]",
        "vmovups ymm2, [rcx]",
        "vfmadd231ps ymm2, ymm0, ymm1",
        "vmovups [rcx], ymm2",
    ],
    "gemm32": [
        "vmovups xmm0, [rsi]",
        "vbroadcastss xmm1, dword ptr [rdx]",
        "vfmadd231ps xmm4, xmm0, xmm1",
        "vbroadcastss xmm1, dword ptr [rdx+4]",
        "vfmadd231ps xmm5, xmm0, xmm1",
        "vbroadcastss xmm1, dword ptr [rdx+8]",
        "vfmadd231ps xmm6, xmm0, xmm1",
        "vbroadcastss xmm1, dword ptr [rdx+12]",
        "vfmadd231ps xmm7, xmm0, xmm1",
        "add rsi, 16",
        "add rdx, 16",
        "sub rcx, 1",
        "jnz gemm32",
    ],
    "index32": [
        "vpgatherdd ymm0, [rsi + ymm1], ymm2",
    ],
    "sew64": [
        "vmovq xmm0, rsi",
        "vpbroadcastq xmm0, xmm0",
        "vmovupd [rdi], xmm0",
    ],
    "fma64": [
        "vmovupd ymm0, [rsi]",
        "vmovupd ymm1, [rdx]",
        "vmovupd ymm2, [rcx]",
        "vfmadd231pd ymm2, ymm0, ymm1",
        "vmovupd [rcx], ymm2",
    ],
    "gemm64": [
        "vmovupd xmm0, [rsi]",
        "vmovddup xmm1, qword ptr [rdx]",
        "vfmadd231pd xmm4, xmm0, xmm1",
        "vmovddup xmm1, qword ptr [rdx+8]",
        "vfmadd231pd xmm5, xmm0, xmm1",
        "vmovddup xmm1, qword ptr [rdx+16]",
        "vfmadd231pd xmm6, xmm0, xmm1",
        "vmovddup xmm1, qword ptr [rdx+24]",
        "vfmadd231pd xmm7, xmm0, xmm1",
        "add rsi, 16",
        "add rdx, 32",
        "sub rcx, 1",
        "jnz gemm64",
    ],
    "index64": [
        "vpgatherqq ymm0, [rsi + ymm1*1], ymm2",
    ],
    "sad16": [
        "movdqu xmm0, [rsi]",
        "movdqu xmm1, [rdx]",
        "psubw xmm0, xmm1",
        "pabsw xmm0, xmm0",
        "phaddw xmm0, xmm0",
        "phaddw xmm0, xmm0",
        "phaddw xmm0, xmm0",
        "pextrw eax, xmm0, 0",
    ],
    "sw8": [
        "movdqu xmm0, [rsi]",
        "movdqu xmm1, [rdx]",
        "paddusb xmm0, xmm1",
        "movdqu xmm1, [rcx]",
        "pmaxub xmm0, xmm1",
        "movdqu [rsi], xmm0",
    ],
    "wmul32": [
        "vmovups xmm0, [rsi]",
        "vmovups xmm1, [rdx]",
        "vmulps xmm0, xmm0, xmm1",
        "vmovups [rdi], xmm0",
    ],
    "bfly32": [
        "vmovups xmm0, [rdi]",
        "vmovups xmm1, [rsi]",
        "vbroadcastss xmm2, dword ptr [rdx]",
        "vmulps xmm1, xmm1, xmm2",
        "vaddps xmm3, xmm0, xmm1",
        "vsubps xmm0, xmm0, xmm1",
        "vmovups [rdi], xmm3",
        "vmovups [rsi], xmm0",
    ],
    "cmul32": [
        "vmovups xmm0, [rdi]",
        "vmovups xmm1, [rsi]",
        "vbroadcastss xmm2, dword ptr [rdx]",
        "vbroadcastss xmm3, dword ptr [rdx+4]",
        "vmulps xmm4, xmm0, xmm2",
        "vmulps xmm5, xmm1, xmm3",
        "vsubps xmm4, xmm4, xmm5",
        "vmulps xmm5, xmm0, xmm3",
        "vmulps xmm0, xmm1, xmm2",
        "vaddps xmm5, xmm5, xmm0",
        "vmovups [rdi], xmm4",
        "vmovups [rsi], xmm5",
    ],
}

ARM = {
    "fill8": [
        "dup v0.16b, w1",
        "st1 {v0.16b}, [x2]",
        "add x2, x2, #16",
        "subs x0, x0, #16",
        "b.hi fill8",
    ],
    "copy8": [
        "ld1 {v0.16b}, [x1]",
        "st1 {v0.16b}, [x2]",
        "add x1, x1, #16",
        "add x2, x2, #16",
        "subs x0, x0, #16",
        "b.hi copy8",
    ],
    "mask8": [
        "ld1 {v0.16b}, [x2]",
        "cmhi v2.16b, v1.16b, v0.16b",
        "bit v0.16b, v3.16b, v2.16b",
        "st1 {v0.16b}, [x2]",
    ],
    "add8": [
        "ld1 {v0.16b}, [x1]",
        "ld1 {v1.16b}, [x2]",
        "uqadd v0.16b, v0.16b, v1.16b",
        "st1 {v0.16b}, [x3]",
    ],
    "sew16": [
        "dup v0.8h, w1",
        "st1 {v0.8h}, [x2]",
        "add x2, x2, #16",
        "subs x0, x0, #8",
        "b.hi sew16",
    ],
    "vwmul16": [
        "smull v2.4s, v0.4h, v1.4h",
        "smull2 v3.4s, v0.8h, v1.8h",
        "sqxtn v4.4h, v2.4s",
        "sqxtn2 v4.8h, v3.4s",
    ],
    "stride16": [
        "ld1 {v0.h}[0], [x1], x2",
        "ld1 {v0.h}[1], [x1], x2",
        "ld1 {v0.h}[2], [x1], x2",
        "ld1 {v0.h}[3], [x1], x2",
    ],
    "redsum16": [
        "addp v0.8h, v0.8h, v0.8h",
        "addp v0.8h, v0.8h, v0.8h",
        "addp v0.8h, v0.8h, v0.8h",
        "umov w0, v0.h[0]",
    ],
    "sew32": [
        "dup v0.4s, w1",
        "st1 {v0.4s}, [x2]",
        "add x2, x2, #16",
        "subs x0, x0, #4",
        "b.hi sew32",
    ],
    "fma32": [
        "ld1 {v0.4s}, [x1]",
        "ld1 {v1.4s}, [x2]",
        "ld1 {v2.4s}, [x3]",
        "fmla v2.4s, v0.4s, v1.4s",
        "st1 {v2.4s}, [x3]",
    ],
    "gemm32": [
        "ld1 {v0.4s}, [x1], #16",
        "ld1r {v1.4s}, [x2], #4",
        "fmla v4.4s, v0.4s, v1.4s",
        "ld1r {v1.4s}, [x2], #4",
        "fmla v5.4s, v0.4s, v1.4s",
        "ld1r {v1.4s}, [x2], #4",
        "fmla v6.4s, v0.4s, v1.4s",
        "ld1r {v1.4s}, [x2], #4",
        "fmla v7.4s, v0.4s, v1.4s",
        "subs x0, x0, #1",
        "b.ne gemm32",
    ],
    "index32": [
        "ldr s0, [x1, x3, lsl #2]",
        "str s0, [x2]",
        "add x3, x3, #4",
        "add x2, x2, #4",
        "subs x0, x0, #1",
        "b.ne index32",
    ],
    "sew64": [
        "dup v0.2d, x1",
        "st1 {v0.2d}, [x2]",
    ],
    "fma64": [
        "ld1 {v0.2d}, [x1]",
        "ld1 {v1.2d}, [x2]",
        "ld1 {v2.2d}, [x3]",
        "fmla v2.2d, v0.2d, v1.2d",
        "st1 {v2.2d}, [x3]",
    ],
    "gemm64": [
        "ld1 {v0.2d}, [x1], #16",
        "ld1r {v1.2d}, [x2], #8",
        "fmla v4.2d, v0.2d, v1.2d",
        "ld1r {v1.2d}, [x2], #8",
        "fmla v5.2d, v0.2d, v1.2d",
        "ld1r {v1.2d}, [x2], #8",
        "fmla v6.2d, v0.2d, v1.2d",
        "ld1r {v1.2d}, [x2], #8",
        "fmla v7.2d, v0.2d, v1.2d",
        "subs x0, x0, #1",
        "b.ne gemm64",
    ],
    "index64": [
        "ldr d0, [x1, x3, lsl #3]",
        "str d0, [x2]",
        "add x3, x3, #8",
        "add x2, x2, #8",
        "subs x0, x0, #1",
        "b.ne index64",
    ],
    "sad16": [
        "ld1 {v0.8h}, [x1]",
        "ld1 {v1.8h}, [x2]",
        "sabd v0.8h, v0.8h, v1.8h",
        "addp v0.8h, v0.8h, v0.8h",
        "addp v0.8h, v0.8h, v0.8h",
        "addp v0.8h, v0.8h, v0.8h",
        "umov w0, v0.h[0]",
    ],
    "sw8": [
        "ld1 {v0.16b}, [x1]",
        "ld1 {v1.16b}, [x2]",
        "uqadd v0.16b, v0.16b, v1.16b",
        "ld1 {v2.16b}, [x3]",
        "umax v0.16b, v0.16b, v2.16b",
        "st1 {v0.16b}, [x1]",
    ],
    "wmul32": [
        "ld1 {v0.4s}, [x1]",
        "ld1 {v1.4s}, [x2]",
        "fmul v0.4s, v0.4s, v1.4s",
        "st1 {v0.4s}, [x3]",
    ],
    "bfly32": [
        "ld1 {v0.4s}, [x1]",
        "ld1 {v1.4s}, [x2]",
        "ld1r {v2.4s}, [x3]",
        "fmul v1.4s, v1.4s, v2.4s",
        "fadd v3.4s, v0.4s, v1.4s",
        "fsub v0.4s, v0.4s, v1.4s",
        "st1 {v3.4s}, [x1]",
        "st1 {v0.4s}, [x2]",
    ],
    "cmul32": [
        "ld1 {v0.4s}, [x1]",
        "ld1 {v1.4s}, [x2]",
        "ld1r {v2.4s}, [x3]",
        "ldr s3, [x3, #4]",
        "dup v3.4s, v3.s[0]",
        "fmul v4.4s, v0.4s, v2.4s",
        "fmul v5.4s, v1.4s, v3.4s",
        "fsub v4.4s, v4.4s, v5.4s",
        "fmul v5.4s, v0.4s, v3.4s",
        "fmul v6.4s, v1.4s, v2.4s",
        "fadd v5.4s, v5.4s, v6.4s",
        "st1 {v4.4s}, [x1]",
        "st1 {v5.4s}, [x2]",
    ],
}

NOTE = (
    "Each comparison table is three assembled strips. "
    "The columns are the ISA, the address, the encoding, and the instruction. "
    "RVV addresses are byte offsets in the acceptance image, and the hex is the instruction word. "
    "Intel bytes are x86-64 in memory order, with no spaces between bytes. "
    "They are assembled by clang with AVX2 and FMA, and without AVX-512, "
    "except \\texttt{wide32} and \\texttt{wide64}, whose Intel strip is EVEX "
    "because that register file is 512 bits. "
    "A splat from a general register is \\texttt{movd} and a shuffle, or \\texttt{vmovq} and \\texttt{vpbroadcastq} from an \\texttt{xmm}. "
    "ARM bytes are AArch64 NEON words, assembled by clang. "
    "SVE is not in these bytes. "
    "An indexed load on NEON is a scalar \\texttt{ldr} and \\texttt{str}. "
    "\\texttt{psubusb} then \\texttt{pcmpeqb} is the unsigned compare. "
    "\\texttt{pcmpgtb} is signed and is not used. "
    "\\texttt{pmullw} alone is the low half of a product. "
    "The \\texttt{vwmul16} Intel strip is \\texttt{pmullw}, \\texttt{pmulhw}, unpack, and \\texttt{packssdw}. "
    "The ARM strip is \\texttt{smull} and \\texttt{smull2}, then \\texttt{sqxtn} and \\texttt{sqxtn2}. "
    "\\texttt{vfmadd231ps} and AArch64 \\texttt{fmla} are fused. "
    "A separate multiply and add is not. "
    "The GEMM strip is one rank-1 update: a vector load, a scalar broadcast, and a fused multiply-add into each accumulator. "
    "\\texttt{vbroadcastss}, \\texttt{vmovddup}, and \\texttt{ld1r} are that broadcast. "
    "\\texttt{wide32} and \\texttt{wide64} are that update on a 512-bit register, four columns, stride \\texttt{vl}. "
    "A 24 by 8 tile is still wider than those four accumulators. "
    "The ARM column of that strip stays NEON. "
    "A label shares its row with the following instruction. "
    "On \\texttt{asum} and \\texttt{iamax} the Intel mask in \\texttt{xmm7} is the absolute-value mask, the same role as the byte mask in \\texttt{mask8}. "
    "The index itself is the RVV \\texttt{vfirst.m}. "
    "\\texttt{quad}, \\texttt{add128}, \\texttt{addd16}, \\texttt{zext64}, and \\texttt{sh64} are RV128I integer strips. "
    "Their first column is labeled RV128. "
    "\\texttt{lq}, \\texttt{sq}, \\texttt{ldu}, and \\texttt{addid} use the decoder placement in Chapter~\\ref{ch:rv128}. "
    "\\texttt{addi} is the ordinary XLEN add. "
    "\\texttt{slli} by 64 uses the seven-bit immediate. "
    "The Intel and ARM columns of those strips keep 64-bit general registers and write the high half explicitly, except \\texttt{quad}, which moves 16 bytes through an \\texttt{xmm} and through NEON. "
    "SVE and AVX-512 are not in those bytes. "
    "\\texttt{sad16} is a halfword absolute difference and a horizontal sum. "
    "\\texttt{pabsw} is the SSSE3 absolute value. "
    "\\texttt{psadbw} sums bytes into words and is a different instruction. "
    "\\texttt{sw8} is \\texttt{paddusb} and \\texttt{pmaxub}. "
    "\\texttt{wmul32}, \\texttt{bfly32}, and \\texttt{cmul32} are "
    "\\texttt{vmulps}, \\texttt{vaddps}, and \\texttt{vsubps}. "
    "Those three are not fused."
)


def assemble(kind, name, lines):
    # The branch target is the kernel name, so that name is the label.
    if kind == "intel":
        src = (
            ".intel_syntax noprefix\n.text\n"
            f".globl {name}\n{name}:\n"
            + "".join(f"  {ln}\n" for ln in lines)
        )
        target = "x86_64-apple-macos"
        flags = (["-mavx512f", "-mfma"] if name in AVX512
                 else ["-mavx2", "-mfma", "-mno-avx512f"])
    else:
        src = (
            ".text\n"
            f".globl {name}\n{name}:\n"
            + "".join(f"  {ln}\n" for ln in lines)
        )
        target = "arm64-apple-macos"
        flags = []
    with tempfile.TemporaryDirectory() as td:
        s = pathlib.Path(td) / "k.s"
        o = pathlib.Path(td) / "k.o"
        s.write_text(src)
        r = subprocess.run(
            ["clang", "-target", target, *flags, "-c", "-o", str(o), str(s)],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"{kind} {name}\n{src}\n{r.stderr}")
        d = subprocess.run(
            ["xcrun", "llvm-objdump", "-d", str(o)],
            capture_output=True, text=True)
        if d.returncode != 0:
            raise SystemExit(d.stderr)
    rows = []
    for line in d.stdout.splitlines():
        m = re.match(
            r"\s+([0-9a-f]+):\s+((?:[0-9a-f]{2} )*[0-9a-f]{2}|[0-9a-f]{8})\s+\S",
            line,
        )
        if not m:
            continue
        addr = int(m.group(1), 16)
        hx = " ".join(m.group(2).upper().split())
        rows.append((f"{addr:04X}", hx))
    if len(rows) != len(lines):
        raise SystemExit(
            f"{kind} {name}: {len(rows)} encodings, {len(lines)} lines\n{d.stdout}")
    return [(a, h, " ".join(asm.split())) for (a, h), asm in zip(rows, lines)]


def rvv_rows(name):
    """One row per encoded instruction. Labels are prefixed later."""
    meta = enc.KERNELS[name]
    words = enc.assemble(meta["ops"])
    rows = []
    for line in enc.listing_lines(name, meta, words):
        left, bar, asm = line.partition("|")
        if not bar:
            continue
        asm = " ".join(asm.split())
        if not asm:
            continue
        if ":" not in left:
            rows.append(("", "", asm))
            continue
        addr, cell = left.split(":", 1)
        rows.append((addr.strip(), cell.strip(), asm))
    return rows


def fold_rows(rows):
    """Prefix each label onto the next instruction that has bytes.

    A register comment is returned separately. A label with no following
    instruction is dropped. Every returned row has a hex cell.
    """
    notes = []
    folded = []
    pending = []
    for addr, hx, asm in rows:
        text = " ".join(asm.split())
        if text.startswith("#"):
            notes.append(text[1:].strip())
            continue
        if not hx.strip():
            if text.endswith(":"):
                pending.append(text)
            continue
        if pending:
            text = " ".join(pending + [text])
            pending = []
        folded.append((addr, hx, text))
    return " ".join(notes), folded


def tex_cell(s):
    if not s:
        return ""
    return f"\\texttt{{{tex_escape(s)}}}"


def tex_escape(s):
    return (s.replace("\\", "\\textbackslash{}")
             .replace("&", "\\&")
             .replace("%", "\\%")
             .replace("_", "\\_")
             .replace("#", "\\#")
             .replace("{", "\\{")
             .replace("}", "\\}")
             .replace("^", "\\textasciicircum{}"))


def emit(name):
    meta = enc.KERNELS[name]
    note, rvv = fold_rows(rvv_rows(name))
    _, intel = fold_rows(
        [("", "", f"{name}:")] + assemble("intel", name, INTEL[name]))
    _, arm = fold_rows(
        [("", "", f"{name}:")] + assemble("arm", name, ARM[name]))
    groups = [("RV128" if name in RV128 else "RVV", rvv), ("Intel", intel), ("ARM", arm)]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Kernel comparison: {name}}}",
        "\\begingroup",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
    ]
    if note:
        lines.append(f"\\noindent {tex_cell(note)}\\par")
        lines.append("\\vspace{0.6ex}")
    lines += [
        "\\begin{tabular}{@{}l|l|l|p{7.2cm}@{}}",
        "\\toprule",
        "ISA & Address & Hex & Assembly \\\\",
        "\\midrule",
    ]
    for i, (isa, rows) in enumerate(groups):
        if i:
            lines.append("\\midrule")
        for addr, hx, asm in rows:
            if isa == "Intel":
                hx = "".join(hx.split())
            if not hx.strip():
                raise SystemExit(f"{name} {isa}: empty hex for {asm}")
            lines.append(
                f"{isa} & {tex_cell(addr)} & {tex_cell(hx)} "
                f"& {tex_cell(asm)} \\\\"
            )
    rel = f"kernels/{meta['folder']}/{enc.fname(name, meta, 'le')}"
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\endgroup",
        "\\end{table}",
        f"\\noindent\\path{{{rel}}}\\par",
        "",
    ]
    return "\n".join(lines)


def add_foreign():
    """Intel and ARM strips for the BLAS bodies and the VLEN-512 tile."""
    pairs = {
        "scal32": (
            [
                "vbroadcastss xmm1, dword ptr [rsi]",
                "vmovups xmm0, [rdi]",
                "vmulps xmm0, xmm0, xmm1",
                "vmovups [rdi], xmm0",
                "add rdi, 16",
                "sub rdx, 4",
                "jnz scal32",
            ],
            [
                "ld1r {v1.4s}, [x2]",
                "ld1 {v0.4s}, [x1]",
                "fmul v0.4s, v0.4s, v1.4s",
                "st1 {v0.4s}, [x1]",
                "add x1, x1, #16",
                "subs x0, x0, #4",
                "b.ne scal32",
            ],
        ),
        "scal64": (
            [
                "vmovddup xmm1, qword ptr [rsi]",
                "vmovupd xmm0, [rdi]",
                "vmulpd xmm0, xmm0, xmm1",
                "vmovupd [rdi], xmm0",
                "add rdi, 16",
                "sub rdx, 2",
                "jnz scal64",
            ],
            [
                "ld1r {v1.2d}, [x2]",
                "ld1 {v0.2d}, [x1]",
                "fmul v0.2d, v0.2d, v1.2d",
                "st1 {v0.2d}, [x1]",
                "add x1, x1, #16",
                "subs x0, x0, #2",
                "b.ne scal64",
            ],
        ),
        "copy32": (
            [
                "vmovups xmm0, [rsi]",
                "vmovups [rdi], xmm0",
                "add rsi, 16",
                "add rdi, 16",
                "sub rdx, 4",
                "jnz copy32",
            ],
            [
                "ld1 {v0.4s}, [x1]",
                "st1 {v0.4s}, [x2]",
                "add x1, x1, #16",
                "add x2, x2, #16",
                "subs x0, x0, #4",
                "b.ne copy32",
            ],
        ),
        "copy64": (
            [
                "vmovupd xmm0, [rsi]",
                "vmovupd [rdi], xmm0",
                "add rsi, 16",
                "add rdi, 16",
                "sub rdx, 2",
                "jnz copy64",
            ],
            [
                "ld1 {v0.2d}, [x1]",
                "st1 {v0.2d}, [x2]",
                "add x1, x1, #16",
                "add x2, x2, #16",
                "subs x0, x0, #2",
                "b.ne copy64",
            ],
        ),
        "axpy32": (
            [
                "vbroadcastss xmm2, dword ptr [rcx]",
                "vmovups xmm0, [rsi]",
                "vmovups xmm1, [rdi]",
                "vfmadd231ps xmm1, xmm0, xmm2",
                "vmovups [rdi], xmm1",
                "add rsi, 16",
                "add rdi, 16",
                "sub rdx, 4",
                "jnz axpy32",
            ],
            [
                "ld1r {v2.4s}, [x3]",
                "ld1 {v0.4s}, [x1]",
                "ld1 {v1.4s}, [x2]",
                "fmla v1.4s, v0.4s, v2.4s",
                "st1 {v1.4s}, [x2]",
                "add x1, x1, #16",
                "add x2, x2, #16",
                "subs x0, x0, #4",
                "b.ne axpy32",
            ],
        ),
        "axpy64": (
            [
                "vmovddup xmm2, qword ptr [rcx]",
                "vmovupd xmm0, [rsi]",
                "vmovupd xmm1, [rdi]",
                "vfmadd231pd xmm1, xmm0, xmm2",
                "vmovupd [rdi], xmm1",
                "add rsi, 16",
                "add rdi, 16",
                "sub rdx, 2",
                "jnz axpy64",
            ],
            [
                "ld1r {v2.2d}, [x3]",
                "ld1 {v0.2d}, [x1]",
                "ld1 {v1.2d}, [x2]",
                "fmla v1.2d, v0.2d, v2.2d",
                "st1 {v1.2d}, [x2]",
                "add x1, x1, #16",
                "add x2, x2, #16",
                "subs x0, x0, #2",
                "b.ne axpy64",
            ],
        ),
        "dot32": (
            [
                "vmovups xmm0, [rsi]",
                "vmovups xmm1, [rdx]",
                "vmulps xmm0, xmm0, xmm1",
                "haddps xmm0, xmm0",
                "haddps xmm0, xmm0",
            ],
            [
                "ld1 {v0.4s}, [x1]",
                "ld1 {v1.4s}, [x2]",
                "fmul v0.4s, v0.4s, v1.4s",
                "faddp v0.4s, v0.4s, v0.4s",
                "faddp v0.4s, v0.4s, v0.4s",
            ],
        ),
        "dot64": (
            [
                "vmovupd xmm0, [rsi]",
                "vmovupd xmm1, [rdx]",
                "vmulpd xmm0, xmm0, xmm1",
                "haddpd xmm0, xmm0",
            ],
            [
                "ld1 {v0.2d}, [x1]",
                "ld1 {v1.2d}, [x2]",
                "fmul v0.2d, v0.2d, v1.2d",
                "faddp v0.2d, v0.2d, v0.2d",
            ],
        ),
        "nrm232": (
            [
                "vmovups xmm0, [rsi]",
                "vmulps xmm0, xmm0, xmm0",
                "haddps xmm0, xmm0",
                "haddps xmm0, xmm0",
                "sqrtss xmm0, xmm0",
            ],
            [
                "ld1 {v0.4s}, [x1]",
                "fmul v0.4s, v0.4s, v0.4s",
                "faddp v0.4s, v0.4s, v0.4s",
                "faddp v0.4s, v0.4s, v0.4s",
                "fsqrt s0, s0",
            ],
        ),
        "nrm264": (
            [
                "vmovupd xmm0, [rsi]",
                "vmulpd xmm0, xmm0, xmm0",
                "haddpd xmm0, xmm0",
                "sqrtsd xmm0, xmm0",
            ],
            [
                "ld1 {v0.2d}, [x1]",
                "fmul v0.2d, v0.2d, v0.2d",
                "faddp v0.2d, v0.2d, v0.2d",
                "fsqrt d0, d0",
            ],
        ),
        "asum32": (
            [
                "vmovups xmm0, [rsi]",
                "vandps xmm0, xmm0, xmm7",
                "haddps xmm0, xmm0",
                "haddps xmm0, xmm0",
            ],
            [
                "ld1 {v0.4s}, [x1]",
                "fabs v0.4s, v0.4s",
                "faddp v0.4s, v0.4s, v0.4s",
                "faddp v0.4s, v0.4s, v0.4s",
            ],
        ),
        "asum64": (
            [
                "vmovupd xmm0, [rsi]",
                "vandpd xmm0, xmm0, xmm7",
                "haddpd xmm0, xmm0",
            ],
            [
                "ld1 {v0.2d}, [x1]",
                "fabs v0.2d, v0.2d",
                "faddp v0.2d, v0.2d, v0.2d",
            ],
        ),
        "swap32": (
            [
                "vmovups xmm0, [rsi]",
                "vmovups xmm1, [rdi]",
                "vmovups [rsi], xmm1",
                "vmovups [rdi], xmm0",
                "add rsi, 16",
                "add rdi, 16",
                "sub rdx, 4",
                "jnz swap32",
            ],
            [
                "ld1 {v0.4s}, [x1]",
                "ld1 {v1.4s}, [x2]",
                "st1 {v1.4s}, [x1]",
                "st1 {v0.4s}, [x2]",
                "add x1, x1, #16",
                "add x2, x2, #16",
                "subs x0, x0, #4",
                "b.ne swap32",
            ],
        ),
        "swap64": (
            [
                "vmovupd xmm0, [rsi]",
                "vmovupd xmm1, [rdi]",
                "vmovupd [rsi], xmm1",
                "vmovupd [rdi], xmm0",
                "add rsi, 16",
                "add rdi, 16",
                "sub rdx, 2",
                "jnz swap64",
            ],
            [
                "ld1 {v0.2d}, [x1]",
                "ld1 {v1.2d}, [x2]",
                "st1 {v1.2d}, [x1]",
                "st1 {v0.2d}, [x2]",
                "add x1, x1, #16",
                "add x2, x2, #16",
                "subs x0, x0, #2",
                "b.ne swap64",
            ],
        ),
        "rot32": (
            [
                "vbroadcastss xmm2, dword ptr [rcx]",
                "vbroadcastss xmm3, dword ptr [rcx+4]",
                "vmovups xmm0, [rsi]",
                "vmovups xmm1, [rdi]",
                "vmulps xmm4, xmm0, xmm2",
                "vfmadd231ps xmm4, xmm1, xmm3",
                "vmulps xmm5, xmm1, xmm2",
                "vfnmadd231ps xmm5, xmm0, xmm3",
                "vmovups [rsi], xmm4",
                "vmovups [rdi], xmm5",
            ],
            [
                "ld1r {v2.4s}, [x3]",
                "add x3, x3, #4",
                "ld1r {v3.4s}, [x3]",
                "ld1 {v0.4s}, [x1]",
                "ld1 {v1.4s}, [x2]",
                "fmul v4.4s, v0.4s, v2.4s",
                "fmla v4.4s, v1.4s, v3.4s",
                "fmul v5.4s, v1.4s, v2.4s",
                "fmls v5.4s, v0.4s, v3.4s",
                "st1 {v4.4s}, [x1]",
                "st1 {v5.4s}, [x2]",
            ],
        ),
        "rot64": (
            [
                "vmovddup xmm2, qword ptr [rcx]",
                "vmovddup xmm3, qword ptr [rcx+8]",
                "vmovupd xmm0, [rsi]",
                "vmovupd xmm1, [rdi]",
                "vmulpd xmm4, xmm0, xmm2",
                "vfmadd231pd xmm4, xmm1, xmm3",
                "vmulpd xmm5, xmm1, xmm2",
                "vfnmadd231pd xmm5, xmm0, xmm3",
                "vmovupd [rsi], xmm4",
                "vmovupd [rdi], xmm5",
            ],
            [
                "ld1r {v2.2d}, [x3]",
                "add x3, x3, #8",
                "ld1r {v3.2d}, [x3]",
                "ld1 {v0.2d}, [x1]",
                "ld1 {v1.2d}, [x2]",
                "fmul v4.2d, v0.2d, v2.2d",
                "fmla v4.2d, v1.2d, v3.2d",
                "fmul v5.2d, v1.2d, v2.2d",
                "fmls v5.2d, v0.2d, v3.2d",
                "st1 {v4.2d}, [x1]",
                "st1 {v5.2d}, [x2]",
            ],
        ),
        "gemv32": (
            [
                "vmovups xmm0, [rsi]",
                "vbroadcastss xmm1, dword ptr [rdx]",
                "vfmadd231ps xmm4, xmm0, xmm1",
                "add rsi, 16",
                "add rdx, 4",
                "sub rcx, 1",
                "jnz gemv32",
            ],
            [
                "ld1 {v0.4s}, [x1], #16",
                "ld1r {v1.4s}, [x2], #4",
                "fmla v4.4s, v0.4s, v1.4s",
                "subs x0, x0, #1",
                "b.ne gemv32",
            ],
        ),
        "gemv64": (
            [
                "vmovupd xmm0, [rsi]",
                "vmovddup xmm1, qword ptr [rdx]",
                "vfmadd231pd xmm4, xmm0, xmm1",
                "add rsi, 16",
                "add rdx, 8",
                "sub rcx, 1",
                "jnz gemv64",
            ],
            [
                "ld1 {v0.2d}, [x1], #16",
                "ld1r {v1.2d}, [x2], #8",
                "fmla v4.2d, v0.2d, v1.2d",
                "subs x0, x0, #1",
                "b.ne gemv64",
            ],
        ),
        "ger32": (
            [
                "vmovups xmm1, [rsi]",
                "vmovups xmm0, [rcx]",
                "vbroadcastss xmm2, dword ptr [rdx]",
                "vfmadd231ps xmm0, xmm1, xmm2",
                "vmovups [rcx], xmm0",
                "add rcx, 16",
                "add rdx, 4",
                "sub rax, 1",
                "jnz ger32",
            ],
            [
                "ld1 {v1.4s}, [x1]",
                "ld1 {v0.4s}, [x3]",
                "ld1r {v2.4s}, [x2], #4",
                "fmla v0.4s, v1.4s, v2.4s",
                "st1 {v0.4s}, [x3], #16",
                "subs x0, x0, #1",
                "b.ne ger32",
            ],
        ),
        "ger64": (
            [
                "vmovupd xmm1, [rsi]",
                "vmovupd xmm0, [rcx]",
                "vmovddup xmm2, qword ptr [rdx]",
                "vfmadd231pd xmm0, xmm1, xmm2",
                "vmovupd [rcx], xmm0",
                "add rcx, 16",
                "add rdx, 8",
                "sub rax, 1",
                "jnz ger64",
            ],
            [
                "ld1 {v1.2d}, [x1]",
                "ld1 {v0.2d}, [x3]",
                "ld1r {v2.2d}, [x2], #8",
                "fmla v0.2d, v1.2d, v2.2d",
                "st1 {v0.2d}, [x3], #16",
                "subs x0, x0, #1",
                "b.ne ger64",
            ],
        ),
        "iamax32": (
            [
                "vmovups xmm0, [rdi]",
                "vandps xmm0, xmm0, xmm7",
                "movhlps xmm1, xmm0",
                "maxps xmm0, xmm1",
                "pshufd xmm1, xmm0, 1",
                "maxss xmm0, xmm1",
            ],
            [
                "ld1 {v0.4s}, [x1]",
                "fabs v0.4s, v0.4s",
                "fmaxv s1, v0.4s",
                "dup v1.4s, v1.s[0]",
                "cmeq v2.4s, v0.4s, v1.4s",
            ],
        ),
        "iamax64": (
            [
                "vmovupd xmm0, [rdi]",
                "vandpd xmm0, xmm0, xmm7",
                "movhlps xmm1, xmm0",
                "maxpd xmm0, xmm1",
            ],
            [
                "ld1 {v0.2d}, [x1]",
                "fabs v0.2d, v0.2d",
                "fmaxp d1, v0.2d",
                "dup v1.2d, v1.d[0]",
                "cmeq v2.2d, v0.2d, v1.2d",
            ],
        ),
        "wide32": (
            [
                "vmovups zmm0, [rsi]",
                "vbroadcastss zmm1, dword ptr [rdx]",
                "vfmadd231ps zmm4, zmm0, zmm1",
                "vbroadcastss zmm1, dword ptr [rdx+4]",
                "vfmadd231ps zmm5, zmm0, zmm1",
                "vbroadcastss zmm1, dword ptr [rdx+8]",
                "vfmadd231ps zmm6, zmm0, zmm1",
                "vbroadcastss zmm1, dword ptr [rdx+12]",
                "vfmadd231ps zmm7, zmm0, zmm1",
                "add rsi, 64",
                "add rdx, 16",
                "sub rcx, 1",
                "jnz wide32",
            ],
            [
                "ld1 {v0.4s}, [x1], #16",
                "ld1r {v1.4s}, [x2], #4",
                "fmla v4.4s, v0.4s, v1.4s",
                "ld1r {v1.4s}, [x2], #4",
                "fmla v5.4s, v0.4s, v1.4s",
                "ld1r {v1.4s}, [x2], #4",
                "fmla v6.4s, v0.4s, v1.4s",
                "ld1r {v1.4s}, [x2], #4",
                "fmla v7.4s, v0.4s, v1.4s",
                "subs x0, x0, #1",
                "b.ne wide32",
            ],
        ),
        "wide64": (
            [
                "vmovupd zmm0, [rsi]",
                "vbroadcastsd zmm1, qword ptr [rdx]",
                "vfmadd231pd zmm4, zmm0, zmm1",
                "vbroadcastsd zmm1, qword ptr [rdx+8]",
                "vfmadd231pd zmm5, zmm0, zmm1",
                "vbroadcastsd zmm1, qword ptr [rdx+16]",
                "vfmadd231pd zmm6, zmm0, zmm1",
                "vbroadcastsd zmm1, qword ptr [rdx+24]",
                "vfmadd231pd zmm7, zmm0, zmm1",
                "add rsi, 64",
                "add rdx, 32",
                "sub rcx, 1",
                "jnz wide64",
            ],
            [
                "ld1 {v0.2d}, [x1], #16",
                "ld1r {v1.2d}, [x2], #8",
                "fmla v4.2d, v0.2d, v1.2d",
                "ld1r {v1.2d}, [x2], #8",
                "fmla v5.2d, v0.2d, v1.2d",
                "ld1r {v1.2d}, [x2], #8",
                "fmla v6.2d, v0.2d, v1.2d",
                "ld1r {v1.2d}, [x2], #8",
                "fmla v7.2d, v0.2d, v1.2d",
                "subs x0, x0, #1",
                "b.ne wide64",
            ],
        ),
        "quad": (
            [
                "movdqu xmm0, [rsi]",
                "movdqu [rdi], xmm0",
                "add rsi, 16",
                "add rdi, 16",
                "ret",
            ],
            [
                "ld1 {v0.16b}, [x0]",
                "st1 {v0.16b}, [x1]",
                "add x0, x0, #16",
                "add x1, x1, #16",
                "ret",
            ],
        ),
        "add128": (
            [
                "mov rax, qword ptr [rsi]",
                "mov rdx, qword ptr [rsi+8]",
                "add rax, 16",
                "adc rdx, 0",
                "mov qword ptr [rdi], rax",
                "mov qword ptr [rdi+8], rdx",
                "ret",
            ],
            [
                "ldr x2, [x0]",
                "ldr x3, [x0, #8]",
                "adds x2, x2, #16",
                "adc x3, x3, xzr",
                "str x2, [x1]",
                "str x3, [x1, #8]",
                "ret",
            ],
        ),
        "addd16": (
            [
                "mov rax, qword ptr [rsi]",
                "add rax, 16",
                "mov qword ptr [rdi], rax",
                "sar rax, 63",
                "mov qword ptr [rdi+8], rax",
                "ret",
            ],
            [
                "ldr x2, [x0]",
                "add x2, x2, #16",
                "str x2, [x1]",
                "asr x3, x2, #63",
                "str x3, [x1, #8]",
                "ret",
            ],
        ),
        "zext64": (
            [
                "mov rax, qword ptr [rsi]",
                "mov qword ptr [rdi], rax",
                "xor edx, edx",
                "mov qword ptr [rdi+8], rdx",
                "ret",
            ],
            [
                "ldr x2, [x0]",
                "str x2, [x1]",
                "str xzr, [x1, #8]",
                "ret",
            ],
        ),
        "sh64": (
            [
                "mov rax, qword ptr [rsi]",
                "xor edx, edx",
                "mov qword ptr [rdi], rdx",
                "mov qword ptr [rdi+8], rax",
                "ret",
            ],
            [
                "ldr x2, [x0]",
                "str xzr, [x1]",
                "str x2, [x1, #8]",
                "ret",
            ],
        ),
    }
    for name, (intel, arm) in pairs.items():
        INTEL[name] = intel
        ARM[name] = arm


def main():
    add_foreign()
    gen = ROOT / "gen"
    gen.mkdir(exist_ok=True)
    names = list(enc.KERNELS)
    parts = ["% Generated by encode_cmp.py. Do not edit.\n"]
    for name in names:
        text = emit(name)
        (gen / f"cmp-{name}.tex").write_text(text, encoding="utf-8")
        print(name, "ok")
    note = (
        "% Generated by encode_cmp.py. Do not edit.\n"
        + NOTE + "\n"
    )
    (gen / "cmp-note.tex").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
