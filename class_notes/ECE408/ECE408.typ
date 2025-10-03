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

== High-Level Architecture of Modern GPUs

#list(
    [Organized into an array of highly threaded #text(weight: "bold")[streaming multiprocessors (SM)]],
    [Each SM has multiple streaming processors (CUDA cores), which share control logic and memory resources],
    [Memory Hierarchy - the shared global memory is DRAM (slow), local memory for each SM is SRAM (fast)]
)

=== Block Scheduling
#enum(
    [Kernel Called],
    [CUDA runtime system launches the grid],
    [Threads are assigned to SMs on a block-by-block basis. All threads in a block are assigned to the same SM. Usually multiple blocks per SM],
    [Limited number of SMs - the runtime system keeps a list of blocks that need to be executed, and when a block finishes execution, a new block is assigned to that SM]
)

=== Barrier Synchronization

CUDA allows threads in the same block to coordinate activity using the barrier synchronization method ```c __syncthreads()```

```c __syncthreads()``` holds a particular thread at the program location of the call (PC) until every thread in the same block reaches that location. 
#text(weight: "bold")[All threads need to be able to reach this program location, and execute ```c __syncthreads()```]

CUDA runtime system ensures all threads have the (memory) resources to arrive at the barrier. 

Threads in different blocks can't perform barrier synchronization, but this is good because it allows the CUDA runtime system to execute blocks in any order relative to each other (thus, programs can scale easily).

=== Warps and SIMD Hardware
As a programmer, one should assume that threads in a block can execute in any order wrt. one another (hence why barrier synchronization is so important).

Once a block is assigned to a SM, it is divided into 32-thread units called #text(weight: "bold")[warps].
#list(
    [Warps are the unit of thread scheduling in SMs],
    [Blocks are partitioned into warps on the basis of thread indices],
    [If a block doesn't have a clean multiple of 32 threads, the last warp is padded with inactive threads],
    [Multidimensional blocks are projected onto a linearized row-major layout before being partitioned into warps]
)

SM implements zero-overhead warp scheduling
#list(
    [Warps are only eligible for execution once all of its operands are ready]
)

Von Neumann Model - A basic computer Architecture
#list(
    [data and programs are stored in the same memory unit],
    [control unit (which has PC, IR), processing unit (ALU, Register File), and I/O]
)

Control units in modern processors are very complex, including fancy fetch logic, separate instruction/data caches, etc.
SMs in GPUs are designed to execute all threads in a warp using SIMD (Single Instruction, Multiple Device)
#list(
    [One instruction is fetched and executed for all threads],
    [Relatively simple control HW compared to CPUs, and its shared across multiple execution units],
    [Shared control units in SIMD designs reult in significantly less power/area costs]
)

=== Control/Branch Divergence

Control Divergence - different threads within a warp taking different branches. This is a disadvantage of SIMD designs.

When faced with control divergence, GPUs use predicated execution, where they sequentially execute both branches.

We can resolve control divergence issues by making branch granularity a multiple of warp size, so that all threads within a warp share control flow.

=== CUDA Memory Model

Memory hierarchy once again: Registers (SRAM) are fast ($~$1 cycle), but few. Main memory is slow ($~$100s of cycles), but huge (GBs or more)

Each Thread can:
#list(
    [R/W per-thread registers ($~$1 cycle)],
    [R/W per-block shared memory ($~$5 cycles)],
    [R/W per-grid global memory ($~$500 cycles, but there are L2/L1 caches which can reduce this)],
    [Read-only per-grid constant memory ($~$5 cycles with caching)]
)

== Matrix Multiplication - Labs 2 & 3

=== Naive Implementation
Assign one thread to each element in the output matrix, read from global memory for each value in the output matrix.

This approach sucks because the global memory bandwidth cannot supply enough data to keep all of the SMs busy. 

Let's assume we have a GPU which has 1000 GFLOP/s of compute power, and 150 GB/s memory bandwidth.
In the naive implementation, each time we write into the output matrix, we perform two FP operations (multiply-add). 
Furthermore, every time we do these two operations, we have to read 8B of memory from global memory (float is 4B). 
Thus, its 4B/FLOP. 

(150 GB/s)/(4B/FLOP) = 37.5 GFLOP/s, which is significantly less than the theoretical maximum of 1000 GFLOP/s

=== Tiled Matrix Multiplication
A better approach at matrix multiplication, which uses shared memory to avoid unnecessary global memory reads.

Keep in mind, shared memory has a much lower latency than global memory!

To declare shared memory within a kernel, use the ```c __shared__``` modifier
#list(
    [example: ```c __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];```]
)

High-level Idea:
#list(
    [Break input matrices into NxN tiles],
    [Read tile into shared memory],
    [Each thread can then read this local tile from shared memory],
    [Repeat until we've computed the output matrix]
)

While implementing tiled matmul, we need to use barrier synchronization to ensure that the shared memory tile has been completely loaded before we procede with computation.
This idea of: \
doing some work #sym.arrow waiting for threads to catch up #sym.arrow repeat \
is called #text(weight: "bold")[bulk synchronous execution] and dominates HPC applications.

The use of large enough shared memory tiles shifts the bottleneck in Matrix-Matrix multiplication. 
ex: Same GPU with 1000 GFLOP/s compute, 150 GB/s memory BW. If we use $16 times 16$ tiles, we reduce global memory accesses by a factor of 16.

Thus, (150 GB/s)/(4B/FLOP) $times$ 16 = 600 GFLOP/s.

If we use $32 times 32$ tiles, we get a theoretical 1200 GFLOP/s, at which point memory bandwidth is no longer the bottleneck.

Shared Memory Limitations
#list(
    [Implementation Dependent],
    [64kB per SM in Maxwell architecture],
    [Ex: tile width of 16 #sym.arrow 256 threads/block #sym.arrow $2 times 256 times 4"B"$ = 2kB of shared memory/block #sym.arrow upper limit of 32 active blocks],
    [However, there is a maximum of 2048 threads/SM, which inherently limits number of blocks to 8.]
)

= Midterm 2
Will update this later...