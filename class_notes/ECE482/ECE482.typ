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
        font: "Libertinus Serif",
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
                #block[#text(size: 30pt)[ECE 482]]
                #block[#text(size: 16pt)[Fall 2026]]
            ])
            #block(height: 100%, align(horizon)[
                #block[#text(size: 16pt)[#notes_title]]
            ])
            #block(height: 50%, align(horizon)[
                #block[#text(size: 16pt)[These notes aren't fully comprehensive.]]
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

#show math.equation.where(block: true): set align(left)

#show: template.with(
    class_name: "ECE 482",
    notes_title: "Review Notes for ECE 482 (Digital IC Design)",
    names: ("Aniketh Tarikonda (aniketh8@illinois.edu)", ""),
)

= Intro & MOSFET Static Analysis
\

- By the textbook convention, we have a positive drain current when it flows into the drain.
  - NMOS drain current should be positive, PMOS should be negative.

- Industry is going towards SiP instead of SoC because of yield issues and reticle sizes making large monolithic dies unlikely

* Technology Scaling *
  - generation is $18 - 24$ months
  - feature sizes shrink $~0.7x$ per generation, $2x$ transistors per area (insert Dennard Scaling mention here)
  - STI (the other one) stands for shallow trench isolation and can prevent leakage between adjacent transistors
  - Moore's Law

* Supply Voltage Scaling *
  - if $V_"DD"$, $L_"gate"$, $T_"ox"$ reduce by an identical factor, the lateral and vertical electric fields stay the same, so performance also stays the same.
  - power is reduced as well
  - historically, $L_"gate"$ was shrunk more aggressively than supply voltage, so transistor feature size scaling virtually stopped at 90nm

*Noise*
  - one of the quality metrics of a digital IC 
  - can impact the robustness/functionality of a digital IC
    - inductive coupling
    - capacitative coupling
    - power and ground noise

  - *VTC* - Voltage-Transfer Characteristics
    - plot of output voltage as a function of input voltage
  
*Important Parts of a VTC Curve* (For an Inverter)
  - $V_"OL"$ - output low 
  - $V_"OH"$ - output high
  - $V_"IL"$ - input low (point where derivative of VTC is $-1$ for the first time)
  - $V_"IH"$ - input high (point where derivative of VTC is $-1$ for the second time)
  - $V_M$ - inverter switching threshold (point where $V_"out" (V_"in") = V_"in"$)

  - $V_"OL"$ and $V_"OH"$ are defined wrt. each other.
  - In the case of the inverter, we can create a flip-flop by hooking the inputs/outputs of two inverters together.
    - there are two stable operating points ($(V_"OH", V_"OL")$ and $(V_"OL", V_"OH")$), and a metastable point 

*Delay*
  - $t_p$ - propagation delay from when $V_"in" = V_"50"$ until $V_"out" = V_"50"$
    - $V_"50" = (V_"OH" + V_"OL")/2$
  - *$t_"pLH" != t_"pHL"$* necessarily

  - One of the ways we can measure delay is through a *ring oscillator*
    - odd number of inverters connected in a circular chain 
    - period T of the oscillation is given by $T = 2 t_p N$ where $N$ is the number of inverters

*Long-Channel Model*

When $V_"gs,n" < V_"T, n"$ or $V_"gs,p" > V_"T, p"$, $I_"ds" = 0$

When $V_"ds,n" < V_"dsat,N"$ or $V_"ds,p" > V_"dsat,p"$, $I_"ds" = mu C_"ox" (W/L) (V_"ov" V_"ds" - 1/2 V_"ds"^2)$

When $V_"ds,n" >= V_"dsat,N"$ or $V_"ds,p" <= V_"dsat,p"$, $I_"ds" = 1/2 mu C_"ox" (W/L) (V_"ov"^2)$

Instead of using $mu C_"ox"$, we use $k'$
  - under our convention, PMOS $k'$ should be negative

*Threshold Voltage*

$V_t = V_"t0" + gamma (abs(sqrt(V_"sb" - 2 phi_F)) - sqrt(abs(-2 phi_F)))$
  - By design $V_"sb" >= 0$ for NMOS and $V_"sb" <= 0$ for PMOS

Short vs. Long Channel
  - short channel enters saturation "earlier" at a lower $V_"ds"$
  - in the saturation region, the slope of the short channel device is higher
  - no apparent quadratic relation with $V_"gs"$ in saturation region.

*Velocity Saturation*
  - carrier velocity increases linearly with E-field only up to a few kV/cm
  - there are a few equations for this, the basic idea being that we define two regions where $E_"lat" < E_c$, and then $E_"lat" = E_c$ where $E_c$ is the critical E-field.

*Non-Ideal Effects*
  - channel-length modulation & DIBL, to account for these, we add $V_"ds"$ related terms to the IV equation.
  - $u_"eff" = f(V_"gs")$, in other words, $T_"ox, eff" != T_"ox, physical"$

#pagebreak()

To account for these short-comings of the Long-Channel Model, we introduce the *Short-Channel Model*.
  - $I_"ds" = k ((V_"gs" - V_t)V_"min" - 1/2 V_"min"^2)(1 + lambda V_"ds")$
  - $V_"min,n" = min(V_"ds", V_"gt", V_"dsat,n")$ or $V_"min,p" = -min(abs(V_"ds"), abs(V_"gt"), abs(V_"dsat,p"))$

- high $V_t$ transistor is often called "low power" because leakage current is very low.
- low $V_t$ transistor is often called "high performance" because ON state current will be higher.

*Subthresold Operation*
  - exponential dependence of $I_"ds"$ on $V_"gs"$ in the subthreshold region.
  - neither long or short channels accurately represent $I_"ds"$ in the subthreshold region.

  - subthreshold $V_t$ is defined as the $V_"gs"$ at which point $I_"d" = 100 "[nA]" * (W/L)$
    - IV plot is linear below this point, assuming you're using a semi-log plot.
    - $I_"d, threshold" = 100(W/L)10^((V_"gs" - V_"t")/S) "[nA]"$
    
  - *there is a parasitic BJT within a MOSFET*, which is where the exponential relation with $V_"gs"$ comes from.

#pagebreak()

= Dynamic Analysis of CMOS Inverter

- If we have a square wave input, on the rising edge of the input:
  - our $V_"ds, n" = V_"dd"$
- After $t_"pHL"$, $V_"ds" = V_"dd"/2$
- Doing KCL on the output node, we get the following equation
$
C_L (d V_"out") / (d t) = k_n' (W/L) [(V_"dd" - V_"tn0")V_"dsatn" - 1/2 V_"dsatn"^2](1 + lambda_n V_"out") \

integral_(V_"dd")^(V_"dd"/2) C_L / I_"dn" dif V_"out" = integral_0^(t_"pHL") dif t = t_"pHL" \

"assuming" (1 + lambda_n V_"out") approx 1 \

I_"dn" approx (W/L) k_n' (...) = (W/L) I_"dsat,n" \

t_"pHL" = (L C_L) / (W I_"satn") integral_(V_"dd")^(V_"dd"/2) dif V_"out" = (L C_L V_"dd") / (2 W I_"satn")
$

Ultimately, we have a similar process when calculating $t_"pLH"$ --the NMOS is in cutoff and the PMOS is in triode/saturation. Thus, we end up with

$
t_"pLH" = (L C_L V_"dd") / (2 W I_"satp")
$

Using an RC model of delay, we end up with

$
t_"pHL" = 0.69 / S R_"n0" C_L \
t_"pLH" = 0.69 / (S beta) R_"p0" C_L \
$

where $S = W_"N"/W_"min"$ and $beta = W_"P"/W_"N"$

== MOS Capacitances

- $C_"drain"$ is the intrinsic (self-loading) capacitance of the MOSFET
- $C_"gate"$ is the fanout/extrinsic capacitance
- $C_"wire"$ is the (somewhat negligible) wire capacitance

- *5 capacitor model*
  - $C_"gb"$, $C_"gd"$, $C_"gs"$, $C_"sb"$, $C_"db"$
  - we don't really care about $C_"sb"$

Because of Miller's theorem, we can view $C_"gd"$ as two separate capacitors (one on input, one on output)

Theres also overlap capacitance (overlap between gate and drain/source)
  - added capacitance form *fringing fields*

#table(
  columns: (1.4fr, 1.4fr, 1.4fr, 1.4fr),
  align: center,
  stroke: 0.8pt,

  [*Operation Region*],
  [$C_"gb"$],
  [$C_"gs"$],
  [$C_"gd"$],

  [*Cut-off*],
  [$C_"ox" W L$],
  [$C_o W$],
  [$C_o W$],

  [*Triode*],
  [$0$],
  [$frac(C_"ox" W L, 2) + C_o W$],
  [$frac(C_"ox" W L, 2) + C_o W$],

  [*Saturation*],
  [$0$],
  [$frac(2 C_"ox" W L, 3) + C_o W$],
  [$C_o W$],
)

=== PN-junction Capacitances

- textbook assumes LOCOS (local oxidation of silicon), which is old, and kind of obsolete for post $250 mu m$ processes.

- $C_"diff" = C_"bottom" + C_"sidewall" = C_J * "Area" + C_"JSW" * "Perimeter" = C_J L_S W + C_"JSW" * (2L_S + W)$
- $C_J = C_"J0" / (1 + V_J/phi_B)^m$ where $V_J$ is the voltage drop across the PN junction, $V_J <= 0$ for drain-body and source-body connection.

*Non-Linear Capacitances*
  - often times, we will need to use average capacitances.
  - $overline(C) = 1 / (V_j - V_i) integral_(V_i)^V_j C(V) dif V$ where $C(V) = C_"J0" / (1 + V/phi_B)^m$
  - we end up with $overline(C) = K_"eq" C_"j0"$