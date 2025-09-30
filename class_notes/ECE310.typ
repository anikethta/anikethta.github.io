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
                #block[#text(size: 30pt)[ECE 310]]
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
    notes_title: "Review Notes for ECE 310 (Digital Signal Processing)",
    names: ("Aniketh Tarikonda (aniketh8@illinois.edu)", ""),
)

= Midterm 1

== DT vs. CT signals

ECE 210 dealt primarily with CT (continuous time) signals. ECE 310 deals with DT (discrete time) signals. \
Signals:
#list(
    [Continuous-domain (analog) $ #sym.arrow x(t) "for" (t in RR)$ ],
    [Discrete-domain (digital) $ #sym.arrow x[n] "for" (n in ZZ)$ ]
)
Notation is technically important, $x[n]$ refers to a singular sample at n. A signal is represented with ${x[n]}_n "for" n in ZZ$
, and a system can be represented as ${y[n]} = S{x[n]} "for" n in ZZ$. However, I really cannot be bothered to do this, and most problems don't either.

== LTI/LSI Systems

Linear Time Invariant or Linear Shift Invariant (same thing). 
#list(
    [Linear: Superposition of inputs has a corresponding superposition of outputs. Stuff like scaling is preserved. ],
    [Time/Shift Invariant: Time shift in the input results in an equal time shift in the output.]
)
Most of the fancy theorems in this class rely on systems being LTI. 

Determining linearity is relatively straightforward (IMO) but time-invariance isn't (again, IMO). But generally the approach seems to be plugging in a time shift 
$n_o$ into the input. Then apply the same time shift in the output, and see if the two expressions match. 

Trivial Example: 
$y[n] = x[n^2]$

$x[n]$ transforms $x[n] #sym.arrow x[n - n_o] $. 
Thus, $y[n] = x[n^2 - 2n n_o + n_o^2]$. Applying the same time shift to the output, we get $y[n - n_o] = x[n^2 - n_o]$.
These are not the same expression, and thus this system is time-variant. 

== The Delta Function
Similar to the delta function from ECE 210. Except its not an infinite spike. 

$delta[n] := cases(1 "if" n = 0,
                0 "otherwise")$

We can represent a DT signal as a superposition of scaled and shifted delta functions.
For example:
#list(
    [$x[n] = sum_(k in ZZ) x[k] delta[n - k]$],
    [$y[n] = sum_(k in ZZ) x[k] S{delta[n - k]}$]
)

== Convolution
Thankfully we don't have to do 3D tiled convolution kernels for this class.

In ECE 210, we use convolution to "apply" an impulse response to an input CT signal. 
In this class its more or less the same, except its with DT signals.

Discrete convolution is defined as the following: \
$y[n] = (x convolve h)[n] = sum_(k in ZZ) x[k] h[n - k] = sum_(k in ZZ) h[k] x[n - k]$ \

Some useful properties of convolution:
#list(
    [$"start point" = ("start of" x[n]) ("start of" h[n]) $],
    [$"end point" = ("end of" x[n]) ("end of" h[n]) $],
    [Convolution is associative, distributive, commutative (and linear)],
)

== LCCDEs
Stands for Linear Constant Coefficient Difference Equations, and is a popular way to represent LTI/LSI systems.
There are two ways of solving LCCDE's: guess-and-check (painful) and Z-transforms (not painful).

General form of LCCDEs:
$
y[n] = sum_(i = 1)^(K) b_i y[n - i] + sum_(j = 0)^(M) c_i x[n - j] \
"for" {K, M in ZZ | 0 <= K < infinity "and" 1 <= M < infinity}
$

I may be a little too ECE 210-pilled, but I guess you can think of the left summation as the Zero-Input terms, and the right summation
as the Zero-State terms.

=== FIR/IIR Systems

FIR (Finite Impulse Response) systems occur when you have LCCDEs with $K = 0$ (no feedback terms).
IIR (Infinite Impulse Response) systems have $K > 0$. 

== Z-Transforms
Motivation for Z-Transforms: Can we find a class of signals which do not change shape once passed through
an LTI/LSI system?

The Z-Transform $X(z)$ of a DT signal $x[n]$ is defined as: \
$ X(z) = sum_(n = -infinity)^(infinity) x[n] z^(-n) "where" z in CC $ \

The ROC (region of convergence) is the range of values in the z-domain in which the Z-transform converge.
#text(weight: "bold")[Any particular Z-transform may have multiple ROCs which correspond to different inverse Z-transforms.]

A Few Reoccuring Z-Transforms:
#list(
    [$alpha^n u[n]$ $#sym.arrow$ $frac(1, 1 - alpha z^(-1))$ (Right-handed signal, ROC: $|z| > alpha$)],
    [$-alpha^n u[-n - 1]$ $#sym.arrow$ $frac(1, 1 - alpha z^(-1))$ (Left-handed signal, ROC: $|z| < alpha$)],
    [$delta[n - m] #sym.arrow z^(-m)$]
)

A Few Reoccuring Z-Transform properties
#list(
    [Linearity],
    [Time Shifting: $x[n - k] #sym.arrow z^(-k)X(z)$],
    [Convolution: $x_1[n] convolve x_2[n] #sym.arrow X_1(z) X_2(z)$. 
    Note: this is a reoccuring property of FTs and LTs as well. Pretty sure thats why a lot of convolution algo's use FFT so that convolution becomes an $O(n log(n))$ operation]
)

Due to the convolution property, we can express LTI systems in the Z-domain as follows:
$
y[n] = (x convolve h)[n] #sym.arrow  Y(z) = X(z)H(z)
$
Alternatively, we can write $H(z) = frac(Y(z), X(z))$ and inverse Z-transform to recover the impulse response $h[n]$.
This is probably the best way of determining the impulse response for LCCDEs (for now, at least).

#text(weight: "bold")[Inverse Z-transforming often devolves into a lot of partial fraction decomposition, so review that. ]

== Causality

Pretty much the same as introduced in ECE 210. 

Output depends solely on current/past inputs #sym.arrow #text(weight: "bold")[Causal] \
Output depends on future inputs #sym.arrow #text(weight: "bold")[Not Causal]

An anticausal system relies #text(weight: "bold")[solely] on future inputs. 

A system is causal if impulse response $h[n] = 0 "for" n < 0$.

Both causal and non-causal systems have their merits. Causal systems are used often in real-time systems. Example application of non-causal systems would be in
image post-processing, and (generally) signal processing on data which has been stored in some memory unit.

== BIBO Stability

A system is BIBO (Bounded-Input Bounded-Output) stable if for some ${y[n]}_n = S{x[n]}_n$, whenever $|x[n]| <= B_("in") < infinity$,
it holds that $|y[n]| <= B_("out") < infinity$. 

Because of convolution properties, a finite length impulse response is indicative of a BIBO stable system.

What if $h[n]$ is infinite-length? Then it must converge. \
$S{h[n]}$ is BIBO stable iff $sum_(k = -infinity)^(infinity) |h[n]| < infinity$.

=== Z-domain and Stability

In general, a system with transfer function $H(z)$ and associated $"ROC"_H$ is stable if it contains $|z| = 1$ (the unit circle).

With problems with multiple terms within the transfer function, the ROC is at least the intersection of all terms.
For problems that ask to find a bounded input which will result in a bounded output for a non-bounded system, use pole-zero cancellation.
For problems that ask to find a bounded input which results in an unbounded output, try to make divergent terms non-zero (delta function does the trick usually).

An LTI system is #text(weight: "bold")[marginally stable] if its ROC is open at the unit circle $|z| = 1$.

= Midterm 2. 

Will update this when midterm 2 rolls around :D

= Midterm 3

Will update this when midterm 3 rolls around :D