#let template(
    class_name: none,
    notes_title: none,
    names: (),
    doc,
) = {
    set page(
        paper: "us-letter"
    )

    set text(
        font: "New Computer Modern",
        size: 12pt,
    )

    set heading(
        numbering: "1.1.1 "
    )

    set math.equation(
        numbering: "(1)"
    )

    show figure: set block(spacing: 2.5em)
    show math.equation: set block(spacing: 2em)

    let cover_page = [
        #grid(
            columns: 1fr,
            rows: (1fr, 1fr, 1fr),
            align: center,
        )[
            #block(height: 100%, align(horizon)[
                #block[#text(size: 30pt)[ECE 408]]
                #block[#text(size: 16pt)[Fall 2025]]
            ])
            #block(height: 100%, align(horizon)[
                #block[#text(size: 16pt)[#notes_title]]
            ])
            #block(height: 50%, align(horizon)[
                #block[#text(size: 16pt)[This is in no way comprehensive.]]
            ])
            #block(height: 100%, align(horizon)[
                #block[#text(size: 16pt)[
                    #names.join("\n")
                ]]
            ])
        ]
    ]

    cover_page
    pagebreak()

    outline(depth: 3)
    pagebreak()

    set page(numbering: "1")
    counter(page).update(1)
    doc
}

#show: template.with(
    class_name: "ECE 310",
    notes_title: "Review Notes for ECE 408 (Applied Parallel Programming)",
    names: ("Aniketh Tarikonda (aniketh8@illinois.edu)", ""),
)

= Midterm 1

== Why do GPUs exist?

Moores Law - observation that number of transistors on ICs double every 18-24 months.
Dennard Scaling - As feature sizes decrease, energy density remains constant and clock speeds increase. 
#list(
    [$P prop C f V^2$ and capacitance C is proportional to area],
    [Exponential increase in clock speed],
    [Increased transistor density meant memory went from being expensive to effectively infinite]
)

=== End of Dennard Scaling

Dennard Scaling ended around 2005/6, clock speeds stagnated, and we needed different methods to achieve performance expectations.
#list(
    [ILP (Instruction Level Parallelism)],
    [Manycore Systems],
    [Specialization, including GPUs]
)

CPUs vs. GPUs
#list(
    [CPUs are latency-oriented (large ALUs, FUs, large caches, branch prediction, data bypassing, out-of-order execution, multithreading to hide short latency)],
    [GPUs are throughput-oriented with many small ALUs, small caches, simple control logic, and massive multithreading capabilities],
    [CPUs wins perf-wise for sequential, latency-heavy code. GPUs win perf-wise for parallelizable, throughput-focused code.]
)

CUDA - Computing Unified Device Architecture

Threads - a PC, IR, and context (registers & memory)
#list(
    [Many threads #sym.arrow context switching becomes inconvienient],
    [we'd like to avoid communication between threads as much as possible]
)

=== Amdahl's Law

$
t := "sequential execution time" \
p := "% parallelizable" \
s := "speedup on the parallelizable part" \

t_"parallel" = (1 - p + frac(p, s)) times t

$

Effectively, the maximum speedup ($frac(t_"sequential", t_"parallel")$) is limited by the fraction of execution that is parallelizable. 

== Basic Organization of CUDA

CUDA integrates the device (GPU) and host (CPU) into one application. The host handles serial/moderately parallel tasks, whereas the device
handles the highly parallel sections of the program. 

CUDA kernels are executed as a grid of threads
#list(
    [All threads in a grid run the same kernel (SIMT)],
    [Each thread has a unique index that can be used to index into memory/make control decisions]
)

In CUDA, threads are organized within blocks
#list(
    [Threads within a block can cooperate via shared memory, barrier synchronization, and atomic operations]
)

Threads within a block are 3D, blocks within a grid are also 3D. 

```c
gridDim.x // gives you # of blocks in grid (in x axis)
blockDim.x // gives you # of threads within a block (x axis)
blockIdx.x // gives you the index of the block within the grid (x axis)
threadIdx.x // gives you the index of the thread within the block (x axis)
```

Host and Device have their own separate memories with some interconnect between them (PCIe, iirc). Thus, for most
programs you have to:
#enum(
    [Allocate GPU memory],
    [Copy data from CPU to GPU memory],
    [Perform computation using GPU memory],
    [Copy data from GPU to CPU memory],
    [Deallocate GPU memory]
)

The ```c __global__ ``` keyword defines a kernel (callable from host/device, but executes on device).
There also exists ```c __host__``` and ```c __device__``` keywords that are callable/executes from host and device, respectively. 
#list(
    [```c __global__``` must return void, but the other two can return non-void]
)

Example: ```c __global__ vecAdd(float* A, float* B, float* C, int n)``` \
To launch this kernel, you can do the following: \
```c vecAdd<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, n);```
where ```c dimGrid``` is the number of blocks per grid, and ```c dimBlock``` is the number of threads per block. 

There exists a ```c dim3``` type in CUDA which makes multidimensional grids/blocks easier to launch.

#text(weight: "bold")[Blocks can be executed in any order.]