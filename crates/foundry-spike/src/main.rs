//! Spike item 0.5.1: prove the toolchain can compile and run a kernel on the GPU.
//! Compiles a vector-add kernel at run time (NVRTC), runs it on device 0, and checks the result on the host.

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

const KERNEL: &str = r#"
extern "C" __global__ void vec_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(KERNEL)?;
    let module = ctx.load_module(ptx)?;
    let vec_add = module.load_function("vec_add")?;

    let n: usize = 1_000_000;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| 2.0 * i as f32).collect();

    let a_dev = stream.clone_htod(&a)?;
    let b_dev = stream.clone_htod(&b)?;
    let mut out_dev = stream.alloc_zeros::<f32>(n)?;

    let n_i32 = n as i32;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut launch = stream.launch_builder(&vec_add);
    launch.arg(&a_dev).arg(&b_dev).arg(&mut out_dev).arg(&n_i32);
    // SAFETY: argument types and order match the kernel signature above.
    unsafe { launch.launch(cfg)? };

    let out = stream.clone_dtoh(&out_dev)?;
    stream.synchronize()?;

    // Every value is an integer below 2^24, so f32 addition is exact: compare bit-for-bit.
    let mismatches = out
        .iter()
        .zip(a.iter().zip(b.iter()))
        .filter(|(o, (x, y))| **o != **x + **y)
        .count();

    println!("elements: {n}");
    println!("mismatches: {mismatches}");
    if mismatches == 0 {
        println!("RESULT: PASS");
        Ok(())
    } else {
        println!("RESULT: FAIL");
        std::process::exit(1);
    }
}
