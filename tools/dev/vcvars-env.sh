# Source this before `cargo build` when a build script invokes nvcc (e.g. candle-kernels
# compiling .cu -> PTX): nvcc needs cl.exe + MSVC/CRT headers on PATH/INCLUDE/LIB, which
# only the VS Developer environment provides. Plain Git Bash / PowerShell lack it.
# (The cudarc dynamic-loading/dynamic-linking builds do NOT need this; only nvcc at
# build time does.)
#
# Usage:  . tools/dev/vcvars-env.sh && cargo build -p foundry-spike
# vcvars-dump.bat lives in this folder (space-free path: Git Bash mangles quoted
# args passed to cmd, so the bat path cannot contain spaces).
_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_dump="$(cmd //c "$(cygpath -w "$_dir/vcvars-dump.bat")" 2>/dev/null | tr -d '\r')"
# PATH must go through cygpath: the dump is Windows-format and bash cannot search it.
_winpath="$(printf '%s\n' "$_dump" | grep -E '^PATH=' | head -1 | cut -d= -f2-)"
[ -n "$_winpath" ] && export PATH="$(cygpath -up "$_winpath")"
# INCLUDE/LIB/etc. stay Windows-format: cl.exe and nvcc consume them, not bash.
eval "$(printf '%s\n' "$_dump" | grep -E '^(INCLUDE|LIB|LIBPATH|CUDA_PATH|CUDA_PATH_V[0-9_]+|WINDOWSSDKDIR|UCRT_VERSION|Platform)=' \
  | awk '{print "export \"" $0 "\""}')"
command -v cl.exe >/dev/null || echo "vcvars-env.sh: cl.exe still not in PATH - check tools/dev/vcvars-dump.bat" >&2
# CUDA 13.4's CCCL headers #error out under MSVC's traditional preprocessor and demand
# /Zc:preprocessor. cudaforge (candle-kernels' builder) has no extra-flags hook, but
# cl.exe reads the CL env var on every invocation - including the ones nvcc spawns -
# so this reaches the .cu -> PTX builds without touching their build script.
export CL="${CL:+$CL }/Zc:preprocessor"
