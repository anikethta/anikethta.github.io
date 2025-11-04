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
        font: "New Computer Modern Math",
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
    names: ("Aniketh Tarikonda", ""),
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
    [$"start index" = ("start index of" x[n]) + ("start index of" h[n])$],
    [$"end index" = ("end index of" x[n]) + ("end index of" h[n])$],
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

Define: $x[n] = a^n u[n] "and" h[n] = b^n u[n] "where" |a| = |b| = 1$. We can thus expand $a = e^(j phi)$ and $b = e^(j theta)$, respectively. The system output $y[n] = sum_(k = -infinity)^(infinity) x[n - k]h[k] = sum_(k = -infinity)^(infinity) e^(j phi (n - k))e^(j theta (k)) u[n - k] u[k]$. We factor out $exp(j phi n)$ and simplify the summation bounds to rewrite this as:

$y[n] = e^(j phi n) sum_(k = 0)^(n)e^(j k (theta - phi)) = e^(j phi n)(frac(1 - e^(j(theta - phi)(n + 1)), 1 - e^(j (theta - phi)))) u[n]$. 

When $phi = theta$, the indetermant expression can be written as:

 $y[n] = e^(j phi n)sum_(k = 0)^n (1) = (n + 1) e^(j phi n) u[n]$. 
 
 We can see this expression is unbounded due to the $(n + 1)$ term. 

When $phi != theta$, y[n] oscillates, but remains bounded. 

#text(weight: "bold")[Thus, in a marginally stable system, only inputs that match at least one unit circle pole of the system will produce unbounded outputs. Otherwise, it will be bounded. ]


= Midterm 2. 

== CTFT and DTFT

Continuous-Time Fourier Transform (CTFT): ${ x(t) }_(t in RR) #sym.arrow.l.r {X_c (Omega)}_(Omega in RR) $ where:

$
X_c (Omega) = integral_(-infinity)^(infinity) x(t) e^(-j Omega t) d Omega
$

Discrete-Time Fourier Transform (DTFT): ${ x[n] }_(n in ZZ) #sym.arrow.l.r {X_d (omega)}_(omega in RR) $ where:

$
X_d (omega) = sum_(-infinity)^(infinity) x[n] e^(-j Omega n)
$

Important Things to keep in mind about *both* CTFT and DTFT:
    - CTFT converts a continuous input into a continuous output. DTFT converts a discrete input into a continuous output.
    - DTFT is a special case of the Z-transform, CTFT is a special case of the Laplace transform. 
    - Both DTFT/CTFT provide a representation of a signal as a linear combination of complex exponentials (which can be thought of as 2D vectors in the complex plane)

DTFT properties
    - $X_d (omega)$ is $2 pi$-periodic
    - *If ${ x[n] }_(n in ZZ)$ is real-valued*, $|X_d (- omega)| = |X_d (omega)|$ and $angle X_d (- omega) = - angle X_d (omega)$ (Magnitude is symmetric, phase is anti-symmetric)
    - Convolution in $n$-domain is multiplication in the $omega$-domain. Thus, LTI systems have a corresponding frequency response $H_d (omega)$
    - Most CTFT properties have an analogue for the DTFT
    - Inverse DTFT: $x(t) = (2 pi)^(-1) integral_(- pi)^(pi) X_d (omega) e^(j omega n) d omega$

=== Frequency Response

For any *stable* LSI system, we have $H_d (omega) = H(z)|_(z = exp(j omega))$ where $H(z)$ denotes the transfer function. 

If we have an input signal $x[n] = exp(j omega_0 n)$, and pass it through a LSI system with frequency response $H_d (omega)$, the output $y[n] = H_d (omega_0)exp(j omega_0 n)$
In other words, signals of the form $A exp(j omega_0 n)$ are eigenfunctions of LSI systems. 

Similarly, $x[n] = cos(omega_0 n + phi) #sym.arrow |H_d (omega_0)| cos(omega_0 n + phi + angle X_d (omega_0))$

== Sampling and ADC/DAC

Idea: We have a *continuous* signal $x_c (t)$, and every $T$ seconds, we take a sample of it. We then end up with a *discrete* signal $x[n] = x_c (n T)$. Our sampling period is $T$, and our sampling frequency is $f_s = 1/T$.

Question: What is the relationship between $X_c (Omega)$ and $X_d (omega)$? In other words, what effect does sampling have in the Fourier domain?

Quick lil' derivation:

By definition, we know  $x[n] = x_c (n T) = (2 pi)^(-1) integral_(-infinity)^(infinity) X_c (Omega) e^(j Omega (n T)) d Omega$. Furthermore, $x[n] = (2 pi)^(-1) integral_(-pi)^(pi) X_d (omega) e^(j omega n) d omega$. 

Thus, $ integral_(-infinity)^(infinity) X_c (Omega) e^(j Omega (n T)) d Omega = integral_(-pi)^(pi) X_d (omega) e^(j omega n) d omega$. Substituting $omega = Omega T$, we get:
$ (1/T) integral_(-infinity)^(infinity) X_c (omega / T) e^(j omega n) d omega = integral_(-pi)^(pi) X_d (omega) e^(j omega n) d omega
$

Lastly, because $X_c$ is $2 pi$-periodic, we can expand the LHS as the following. #footnote[
Note: Omitted the shift of $omega$ in the complex exponential term because a shift by $2 pi k$ in the exponent corresponds to a 360 degree rotation (thus leaving the complex term unchanged).]

$
(1/T) integral_(-infinity)^(infinity) X_c (omega / T) e^(j omega n) d omega = 1/T sum_(k in ZZ) integral_(-pi + 2 pi k)^(pi + 2 pi k) X_c ((omega + 2 pi k) / T) e^(j omega n) d omega
$

Using the change of variables $omega' = omega - 2 pi k$, we can further simplify the bounds of the integral expression. Matching the integrands of this expanded expression and the DTFT integral, we determine:

$
  X_d (omega) = 1/T sum_(k in ZZ) X_c ((omega + 2 pi k)/ T)
$

=== Aliasing and Nyquist Frequency

Suppose we have a *band-limited* signal whose Fourier trnasform is $X_c (Omega)$. Thus, $X_c (Omega) = 0$ for $|Omega| > B$,  where $B$ is the bandwidth (highest-frequency) of the signal. 

If we sample this signal, we get that $X_d (omega) = 1/T sum_(k in ZZ) X_c ((omega + 2 pi k)/ T)$. Graphically, this looks like an infinite series of scaled & shifted versions (aliases) of $X_c (Omega)$, each of which centered at some $2 pi k$. To find where the bandwidth gets "mapped" to after sampling, we use the relation $omega = Omega T$ where $Omega #sym.arrow B$. 

We know B is highest-frequency in our original signal, and has units rad/s, so $B = 2 pi f_(B)$. We also know $T = 1/f_s$. Thus, we can express $omega_B$ (the bandwidth of the signal *after* sampling) as $omega_B = B T = 2 pi f_B/f_s$

Each of the aliases is centered at $2 pi k$ for $k in ZZ$. They have the potential to overlap if the bandwidth of each alias in the $omega$-domain exceeds $pi$ (think about it graphically). This overlap is called *aliasing*, and it causes us to lose information about the original signal.

Thus, if we don't want aliasing to occur, the bandwidth in the $omega$-domain must be less than or equal to $pi$.

Equivalenty, $omega_B =  2 pi f_B/f_s <= pi$ #sym.arrow *$f_s >= 2 f_B$*

This condition is called the Nyquist Criterion, and the *Nyquist Frequency* is $2 f_B$, which is the lowest sampling frequency in which you will observe no aliasing. 

In general there are two ways to resolve aliasing for ADC conversion:
 - Increase $f_s$ to at least Nyquist Frequency.
 - Use a low-pass filter (LPF) on $X_c (Omega)$ to artificially reduce $f_B$. #footnote[This comes at a cost of losing information about the original signal. However, often times the information lost by low-passing is much less than the potential information lost by aliasing. In other scenarios however, we can actually just allow aliasing to occur, and then use a digital LPF in the $omega$-domain to select the range of frequencies we need. There was a homework problem about this.]

=== Ideal ADC and DAC

The process of sampling converts a CT signal to a DT one, and so that process is called Analog-to-Digital Conversion (ADC).

Ideal Digital-to-Analog conversion basically consists of two steps:
- Apply a LPF to remove the higher-frequency aliases, and just retain the frequency spectrum centered at $omega = 0$.
- Properly scale/resize the resulting frequency spectrum to obtain that of $X_c (Omega)$

== DFT 

The DTFT is continuous in the Fourier domain, wheras the DFT is discretized in the Fourier domain. 

Given a length-N signal ${x[n]}_(n=0)^(N - 1)$, the N-point DFT ${X[k]}_(k=0)^(N - 1)$ is defined:

$
  X[k] = sum_(n = 0)^(N - 1) x[n] e^(-j (2 pi n k)/N) = sum_(n = 0)^(N - 1) x[n] W_N^(- k n) = sum_(n = 0)^(N - 1) x[n] e^(- j omega n)|_(omega = (2 pi k )/N)
$

Won't come up on the midterm (probably), but the DFT allows for some really cool matrix/vector representations in Fourier Analysis, as it is a linear transformation. 

=== DFT properties

DFT properties are pretty similar to DTFT, albeit with some modifications. 
Here are a few of the important ones:
- DFT is periodic by $N$. Thus, $X[k + a N] = X[k]$,  $a in ZZ$
- $x[angle.l n - m angle.r_N] #sym.arrow exp(-j (2 pi m k)/N) X[k] = W_N^(-k m) X[k]$
- $x[n] #sym.ast.op.o h[n] = X[k] H[k]$

=== Zero-Padding

Idea: if we 'artificially' increase the length of the input signal by adding zeros to the end of it, the DTFT is unchanged. However the DFT, which has the same number of samples as the input signal, increases in length.
More specifically, it takes more "samples" of the DTFT, yielding a higher resolution of the frequency spectrum of the original signal. 

This is important to keep in mind when doing spectral analysis, and we need an increased resolution to better identify peaks. 

=== Windowing

Windowing is a method to attenuate spectral leakage, which is a consequence of the DFT. 

Windowing is just multiplication in the time domain. The two windows we use for this class are Rectangular (boxcar) windows and Hamming windows. 
Rectangular window has a narrower main-lobe frequency response, but larger side-lobes. The Hamming window has a wider main-lobe, but significantly attenuated side-lobes in its frequency response. 
= Final

Will update this when midterm 3 rolls around :D