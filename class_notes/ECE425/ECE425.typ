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
                #block[#text(size: 30pt)[ECE 425]]
                #block[#text(size: 16pt)[Spring 2026]]
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

#show: template.with(
    class_name: "ECE 425",
    notes_title: "Review Notes for ECE 425 (Intro to VLSI System Design)",
    names: ("Aniketh Tarikonda (aniketh8@illinois.edu)", ""),
)

= Intro

- the semiconductor market is growing, and thus we need VLSI people (literally entirety of lecture 1)

- (Brief) History of Computers:
    - Until the $20$th century, we had mechanical computers, abacuses, etc.
    - Vacuum tubes are invented, and can implement boolean logic. First computer is built with vacuum tubes.
    - Vacuum tubes are replaced by transistors
    - Transistors decreased in size, complexity of systems increase
    - ICs invented, can print transistors through lithography. Early LSI era.
    - Technology improves #sym.arrow higher resolution #sym.arrow hundreds of billions of transistors on chips. VLSI era.

- Modern chips use CMOS (complementary nMOS and pMOS networks) to implement digital logic. 

*Obligatory Moore's Law and Dennard Scaling Mention* rahh I can never escape it
    - Dynamic MOSFET Power Consumption: $P_"dyn" = n C V^2 f_"clk"$
    - If dimensions of the transistor scale by $~0.7x$, area scales by roughly a half.
        - Capacitance ($C$) and Voltage ($V$) scale linearly wrt. dimensions, $f_"clk"$ scales by $1/0.7$
        - $2x$ transistors in the same area
        - End result is that scaling dimensions has #emph[no] effect on $P_"dyn"$
        - This observation was the basis for *Dennard Scaling*
    - Dennard Scaling ended around 2005/2006
    
- Economies of Scale, increasing R&D costs for fabrication lead to companies outsourcing fabrication to certain specialized companies (e.g. TSMC, GlobalFoundries, etc.)
    - Rise of EDA industry and global standards for semiconductors (e.g. GDSII)
    - Tools, libraries, PDKs, etc.

- Semiconductors have short market windows, short product life cycles, stiff competition
    - Certain chips need to be low cost, some need to have really good power efficiency

- Modern ASIC/Chip Design Workflow: Design #sym.arrow Architecture #sym.arrow RTL #sym.arrow Gate-Level Netlist #sym.arrow Physical Design (floorplanning, layout, pnr) #sym.arrow fab does their thing, we do Post-Silicon Validation

#pagebreak()

= Midterm 1

== Intro to MOS Transistors
    - We build chips out of silicon because its a semiconductor, has four valence electrons and can form nice crystal lattices (covalent bonds) , has nice thermal properties, and is relatively abundant.
    - We can dope it with an element that has 3/5 valence electrons so that we introduce holes and electrons which travel around the lattice (doping)
        - Doping is generally done via diffusion: exposing silicon to superheated phosphorus/boron gas.
    - Electrons drifting #sym.arrow Current, improves conductivity
    - *n-type semiconductors have extra electrons*
    - *p-type semiconductors have extra holes (lack of electrons)*

    *PN Junctions*
        - a P-N junction forms a diode
        - Initially the electrons in the n-type fill the holes near the junction, forming a *depletion layer*
        - When we put a higher electric potential on the anode (p-type side), current will flow (assuming its greater than the threshold voltage)
            - This is called *forward biasing*
        - Alternatively, we can *reverse bias* the PN junction, which causes the depletion layer to grow and current to stop flowing.

    *nMOS transistors* (invert p and n for pMOS)
        - depletion layer forms around n-wells
        - only have current when p-substrate has a higher potential
        - four terminal device: gate, source, drain, body
        - When gate voltage increases beyond a threshold:
            - An inversion region forms under the gate with electrons as charge carriers
            - Creates an n-channel: current flows from source to drain.
        - We want to keep n-well at a higher potential than the p-type substrate (otherwise we forward bias it)
            - Body connection for nMOS is to GND (reverse biased)

    - Holes move slower than electrons#footnote[Technically holes are just the lack of electrons. In any case, this is because holes "travel" in the valence band, whereas electrons travel in the conduction band.] by a factor of $2-3$x, which is why pMOS transistors are usually sized $~2-3$x larger than nMOS.

    - NMOS passes logical 0 well, passes a degraded 1. PMOS passes logical 1 well, but degraded 0.

    - *CMOS* - combination of PMOS and NMOS
        - pull-up network (PUN) of pMOS, pull-down network (PDN) of NMOS
        - if PUN and PDN are both on, we have a short circuit. 
        - if PUN and PDN are both off, the output is floating (high-Z)

    - PUN is the logical complement of PDN
    - *Demorgan's Law*
        - $(A' + B') = (A B)'$
        - $(A' B') = (A + B)'$

== Intro to Layout 
\
    *Layout Design Rules*
    - this is very idealized because we're using a relatively ancient process node (FreePDK45nm)
        - modern fabrication processes are #emph[significantly] more complex
    
    === Lambda ($lambda$) Design Rules
        - $lambda$ coresponds with half of the minimum feature size
        - feature size is the minimum transistor channel length, or the minimum width of the polysilicon wire
        - allows easy scaling for different (old) processes.
        - not applicable to modern (sub $90$nm) processes

        *Rules: *
        + metal and diffusion have minimum width and spacing of $4 lambda$
        + contacts are $2 lambda times 2 lambda$, surrounded by $lambda$b on layers above and below
        + polysilicon width is $2 lambda$
        + polysilicon and contacts have spacing of $3 lambda$ from others 
        + polysilicon overlaps by $2 lambda$ where it is desired, spacing of $lambda$ away from areas where no transistor is desired
        + n-well surrounds pMOS by $6 lambda$, avoids nMOS by $6 lambda$

    === Guides for Optimized CMOS Layouts
        - optimize boolean expression before drawing stick diagram, layout
        - horizontal $V_"DD"$ rail on top, GND rail on bottom
            - p-diffusions close to $V_"DD"$
            - n-diffusions close to GND
        - minimize metal lengths 
        - polysilicon lines are high-$Omega$, and should generally run vertically. Avoid turns
        - *merge diffusions*
            - e.g. NAND gate, drain and source of PUN can be merged. the drains of PDN can also be merged.
            - large savings on area, routing, performance
        - size transistors appropriately such that equivalent resistances remain somewhat minimized

        *Gate Layout with Euler Paths*
            - draw schematic
            - find Euler path (doesn't have to end at the starting point)
            - ensure label/ordering is the same for PUN/PDN
            - If you do this correctly, you can create designs with nice, straight polysilicon.
            - can be not-so-trivial at times, this is an NP-hard problem

        - Avoid multiple metal layers, if possible (leave room for routing)
        - Don't forget metal-poly, metal-metal, metal-diffusion contacts

        *Standard Cells*
            - Create basic, uniform "building blocks" (NAND, NOR, MUX, DFF, etc.). Lay them out on parallel $V_"DD"$ and GND rails, and then route connections between them.

            - Standard Cells vs. Manual Layout
                - Standard Cells are pre-verified and characterized, allowing them to be used for fast physical design flows using automated synthesis and PnR tools.
                    - Cells are fixed, cannot tweak them to squeeze out better PPA.
                - Manual Layout allows more potential for optimization, but is a lot more painful.
                    - You only want to manually lay out components of your design that require high performance (ex: ALU)

    === Common Combinational and Sequential Circuit Elements

    - Most are self-explanatory, really basic
        - AOI22: "and - or - invert" ($Y = ~ ((A & B) | (C & D))$)
            - AOI21 just passes C (no 2nd NAND gate)
        - OAI22: "or - and - invert" ($Y = ~ ((A | B) "&" (C | D))$)
            - OAI21 just passes C (no 2nd NOR gate)

    *Non-restoring Transmission Gate*
        - nMOS and pMOS in parallel
        - called "non-restoring" because output voltage isn't being driven by $V_"DD"$ or GND
            - signal slowly gets degraded as you put many non-restoring gates in series

    *Tri-States*
        - a transmission gate is one way to build a (non-restoring) tri-state, when $"EN" = 0$, the output is high-Z
        - A Restoring Tri-State Inverter uses two pMOS and two nMOS in series, outputs are directly driven by $V_"DD"$ and GND

    *Multiplexers*
        - can be built via NAND/NOR/AOI22 gates, but not very optimal
        - can be built with two transmission gates (non-restoring)
        - can be build with a pair of tri-state inverters

        - Larger muxes (e.g. 4-1, 8-1) can be build hierarchically using 2-1 muxes, or flattened (4 or 8 tristates)

    - With multiplexers, inverters, and tri-states, you can build sequential elements such as D-latches.
    - By placing two D-Latches in series with an inverter between their $"CLK"$ inputs, we create a DFF (posedge-triggered FF)

    - Back-to-Back DFFs can malfunction due to clock skew, race conditions 
        - Thus, we can insert buffers/gates to add some combinational delay between the DFFs #footnote[Don't overdo this or we end up with setup time failures]

    *Quick Note on Sizing*
        - Transistor network should be sized for equivalent PUN and PDN resistance.
            - Because of different electron/hole mobilities, pMOS should be sized at 2-3x (for ECE425 its 2x) that of an equivalent nMOS.

        - ALl Paths to the output should have equivalent resistance of $R$ or lower.
            - Two unit transistors in series have resistance $2R$
                - To ensure equivalent resistance of $R$, size the gates of both of these transistors at $2x$ the regular amount.
            - Two unit transistors in parallel have resistance $R$ #footnote[You need to consider worse case scenarios where only *one* of the parallel transistors is on at a time]
#pagebreak()

== MOS Transistor Theory
    - Naming: "Source" of a MOSFET is the source of majority charge carriers (electrons in nMOS, holes in pMOS)
    - Three operation modes: Accumulation, Depletion, inversion
        - Accumulation: $V_g - V_b < 0$
        - Depletion: $0 < V_g - V_b < V_t$
        - Inversion: $V_g - V_b > V_t$ 
    - Three Regions in the Shockley (wild ass wiki article btw) Model: Cutoff, Linear, Saturation

    - When $V_"gs" < V_t$, the device is in cutoff (no channel forms)
    - Channel forms when $V_g$ exceeds channel voltage $V_c$ by at least $V_t$
        - $V_c approx V_s$ near source, $V_"gs" = V_g - V_s > V_t$, and so the n-channel forms
        - Near the drain, $V_c approx V_d$, and so $V_"gd" = V_"gs" - V_"ds" < V_t$, and so the n-channel is pinched off

    - In saturation $V_"ds" > V_"ov"$, current will still flow, but theres no dependence on $V_"ds"$ #footnote[$V_"ov"$ is the overdrive voltage, defined as $V_"gs" - V_t$ in nMOS (or $V_"sg" - |V_t|$ in pMOS)]

    *Deriving the Ideal MOSFET Equation(s)*
        - Overview: MOS structure is effectively a parallel plate capacitor (important thing to keep in mind for later discussions on timing/delay)
        - How can we derive the amount of charge in the channel, and how long each charge takes to cross it?

        - Capacitor Eqn: $Q_"charge" = C_g V$ where $C_g$ is the gate capacitance which we can derive using the parallel plate capacitor eqn:
        - $C_g = epsilon_"ox" frac(W L, t_"ox") = C_"ox" W L$. \
            - $epsilon_"ox"$ is the permittivity of the gate oxide, $t_"ox"$ is the gate thickness, $W L$ is the area (width times length)
        - Gate voltage $V = (V_g - V_c) - V_t$. 
            - We can use the approximation $V_c approx frac(V_s + V_d, 2)$ to simplify this to $V = (V_"gs" - V_"ds"/2) - V_t$

        - With the gate voltage/capacitance, we can approximate the charge that forms under the gate. However, $I = frac(Delta Q, Delta t)$, so we also need to derive this $Delta t$.

        - Electric Field (lateral) is $E = frac(V_"ds", L)$ where $L$ is channel length. We also know $v_"electron" = mu E$ where $mu$ is electron mobility.
            - time required for an electron to cross the channel is $t = L/v_"electron"$
            - $Delta t = L/v = frac(L^2, mu V_"ds")$
            
        - With these two components, we can now derive the MOSFET IV-equation for an nMOS in the linear region (Shockley model)
            - $ I_"ds" = mu C_"ox" (W/L) (V_"ov" V_"ds" - 1/2 V_"ds"^2) $
        - To calculate the current at saturation, we plug in $V_"ds" = V_"ov"$, at which point we get:
            - $I_"ds" = 1/2 mu C_"ox" (W/L) (V_"ov"^2)$

    - How do we increase $I_"ds"$?
        - Increase amount of charge in the channel
            - Increase $t_"ox"$ (not in your control)
            - Increase $L$ (bad idea, charges have to cross a longer distance)
            - Increase transistor width $W$ (this is a good idea)
            - Increase gate voltage (also a good idea, *although there are caveats*)
        - Decrease time to cross channel
            - Decrease $L$ (this is intrinsically limited by your minimum feature size)
            - Increase $V_d$/$V_"ds"$

    === Non-Ideal Effects
        - There are two E-fields in a MOSFET
            - a vertical one $E_"vert" = V_"gs"/t_"ox"$ 
            - a lateral one $E_"lat" = V_"ds"/L$

        - Electrons are attracted upwards due to $E_"vert"$, directed them right towards the gate oxide
            - They collide/scatter off of the substrate-oxide interface, degrading velocity (lowering mobility)
            - Increase total drain-to-source path

        - To account for $E_"vert"$, we introduce $mu_"eff" < mu_e$
            - $mu_"eff"$ is determined by best-fitting experimental data.
            - $E_c = 2v_"sat"/mu_"eff"$
         $ v = cases(frac(mu_"eff" E, (1 + E/E_c)) &quad E < E_c , 1 &quad E >= E_c) $

        *Channel Length Modulation*
            - A reversed-biased PN-junction forms a depletion region
                - Width of depletion region $L_d$ grows when when we further reverse-bias it
            - Thus, the channel length is actually smaller when we take this into account:
                - $L_"eff" = L - L_d$, shorter $L_"eff"$ means higher current
                - Increasing $V_"ds"$ past saturation grows the depletion region, thus increasing $L_d$, thus resulting in a shorter $L_"eff"$, which results in a higher current.

            - To account for this, we add a term to the saturation equation to linearly approximate this contribution:
            $ I_"ds" = 1/2 mu C_"ox" (W/L) (V_"ov")^2 (1 + lambda V_"ds") $

        *Threshold Voltage Effects*
            - Depends on body voltage, drain voltage, channel length 
            - intrinsically depends on material, doping concentration, oxide geometry, temperature, etc.
            - Body Effect: $V_t = V_"t0" + gamma (sqrt(phi_s + V_"sb") - sqrt(phi_s))$
                - $gamma$ is the body coefficient, $phi_s$ is the fermi Potential.
            - Drain-Induced Barrier Lowering (DIBL)
                - high $V_d$ creates a large depletion region
                - depletion region spills over to the channel
                - easier for gate to create inversion layer in the channel
                - Another linear approximation: $V_"t'" = V_t - eta V_"ds"$ (threshold voltage decreases)

        *Leakage Currents*
            - *Subthreshold Leakage*: Current flowing from source to drain when in cutoff
                - Caused by weak inversion layers, short-channel effects, thermal agitation
                - The primary leakage concern for lowly students in ECE425
            
            $ I_"ds" = I_"ds0" exp[ (V_(g s) - V_(t 0) + eta V_(d s) - k_gamma V_(s b)) / (n v_T) ] (1 - exp[ (-V_(d s)) / v_T ]) $

            $ I_"ds0" = beta v_T^2 exp[1.8] $

            - Gate Leakage: current flowing between gate and body
                - Charges can quantum tunnel through the gate oxide
            
            - Junction Leakage: current flowing between source and body, drain and body
                - Diode Leakage: fast/heated electrons can cross the depletion region
                - Band-To-Band Tunnel (BTBT): electrons can tunnel through the PN-junction.
                - gate-induced drain leakage (GIDL): high $V_d$, low $V_g$ causes pronounced BTBTn beneath gate overlap 

        Temperature Effects
            - Higher temperature causes reduced mobility, lower $V_t$
            - Electrons now have more energy and are more likely to tunnel
            - Exacerbates most of the other non-ideal effects
            - This is why you should keep your chips relatively cool (or at least be smart with packaging)

        *Parameter Variation*
            - We can't expect transistors to behave ideally
            - "Fast Assumption" : $L_"eff"$ is short, $V_t$ is low, $t_"ox"$ is thin.
            - "Slow Assumption" : $L_"eff"$ is long, $V_t$ is high, $t_"ox"$ is thick.
            - "Typical Assumption" is in the middle of these two.
            - We need to ensure our design works for all of these assumptions
            - *Process Corners* are a way of graphically representing this parameter variation
    === DC Transfer Characteristics of the Inverter
        - Overview: We can derive the current through both the nMOS and pMOS in series, in the inverter configuration. By KCL, the sum of these currents at the output node is $0 A$.
            - We solve this KCL analytically. In the linear region, the current is a function of $V_"ds"$, where $V_d$ is equal to $V_"out"$
            - We can also solve this graphically, finding the intersection of the nMOS and pMOS curve (the pMOS curve needs to be shifted and reflected across x-axis)
        - There are five regions of operation for the CMOS inverter.
        - The ratio of $beta_p/beta_n$ affects the shape of the DC transfer curve.
            - Called "skewed" when $beta_p/beta_n != 1$

        - *Noise Margins*: how much noise can a gate take before it stops producing proper outputs. 
            - The ranges which define the noise margin are generally defined at the unity gain points of the DC transfer curve (aka when the slope of the curve is $-1$)
            - Anything within the range is called an "acceptable" $1$ or $0$.
        - Because the CMOS inverter is a restoring gate, it converts an "acceptable" input into a strong/clean $0$ or $1$

        - Beware of using pass transistors in series
            - pass transistors may be acceptable for driving restoring gates, but not for other pass transistors.
    
    === Timing and Delay
        - All of the previous functions assume inputs/outputs instantaneously reach steady-state. In reality, $V_"in"$ and $V_"out"$ are functions of time. 
        - MOSFET gates act as a parallel plate capacitor, and so charges need to accumulate at the gate in order to switch it ON/OFF.
            - Also takes time to discharge
        - If you have a gate output with high fanout (driving many different input gates), the overall capacitance increases #sym.arrow time to charge/discharge increases #sym.arrow delay increases
= Midterm 2
== Timing and Delay
=== Logical Effort, Elmore Delay, Sizing
- We can model a MOSFET with RC circuits for timing purposes
    - unit nMOS ($k = 1$) has resistance $R$ and capacitance $C$
    - unit pMOS ($k = 2$) has resistance $2 R/2$ and capacitance $2C$
    - $k$ refers to a multiple of the unit size transistor width
    - Resistance is inversely proportional to width, Capacitance is proportional to it.
    - Using this model, we can approximate delay as $tau = R C$ (*RC Delay Model*)
    - worst-case rising/falling paths through comb. logic is called *propagation delay*, best-case is called *contamination delay*

- *Elmore Delay*
    - provides a way to estimate delay for higher-order systems
    - Steps:
        1. for each capacitor, draw the path to VDD (if pull-up) or VSS (if pull-down), and take the sum of the resistances on that path.
        2. for each capacitor in (1), add its delay in the pull-up/pull-down network.
    - Layout affects delay (wow how surprising)
        - Merging diffusions reduces capacitance (separate diffusions doubles capacitance)
        - Large contacts increase diffusion area and thus capacitance. Folded gates reduce capacitance.
    - If a gate is driving other gates, the propagation and contamination delay includes the gates own capacitance (from diffusions) as well as the gate capacitance of the other gates it drives.

- Logical Effort tries to express delay in process-independent units.
    - Effort Delay has two components: Logical Effort and Electrical Effort. $f = g h$
    - Logical Effort: $g = C_"in" / C_"inv"$. $g = 1$ for unit inverter.
    - Electrical Effort (fanout): $h = C_"out" / C_"inv"$
    - delay $d = f g + p = (C_"out" + C_"parasitic") / C_"inv"$
    - Delay across a multistage logical network is minimized when each stage has the same effort.
        - $f = g_i h_i = F^(1/N)$
        - Minimum delay of an $N$-path stage is $N F^(1/N) + P$
- *Sizing*
    - If an input isn't critical (not on the critical path) we cna prioritize other inputs at its expense.
    - *Skew Gates* - favoring one transition over the other.
        - *HI-skew* - favors rising transition
        - *LO-skew* - favors falling transition
        - rising and falling logical effort should be done with respect to the unskewed inverters with equal drive for rising/falling transitions.
            - e.g. HI-skewed inverter with pMOS size 2, nMOS size 0.5 (size (2, 0.5)) should have an rising unskewed inverter which is size (2, 1), whereas the falling unskewed inverter should have size (1, 0.5) 
=== Timing
    
*Terminology*
    - $t_"pd"$ - logical propagation delay (worst-case)
    - $t_"cd"$ - logical contamination delay (best-case)
    - $t_"pcq"$ - FF clock-to-Q propagation delay (how long does it take for Q to take the value of D on edge of clock)
    - $t_"ccq"$ - FF clock-to-Q contamination delay
    - $t_"pdq"$ - FF D-to-Q propagation delay (equivalent of pcq for latches, as long as clk is high, Q will take on the value of D)
    - $t_"cdq"$ - FF D-to-Q contamination delay
    - $t_"setup"$ - we need to hold data for some time before the clock edge
    - $t_"hold"$ - we need to hold data for some time after the clock edge

*Hold Time and Setup Time, Slack*
    - *Setup Time* - Assume a setup of two FFs and a comb. logic cloud between the two.
        - The data from the first FF should be able to "leave" the FF, pass through the comb. logic, and be able to get to the second FF at least $t_"hold"$ before the next clock edge.
        - $T_"req" = T_"clk" - T_"setup"$
        - $T_"arr" = T_"pcq" + T_"pd"$
        - $"setup slack" = T_"req" - T_"arr"$ *we want this to be positive*

    - *Hold Time* - Same setup as above. 
        - We want to hold the input of the FF for some time after the clock edge to ensure that the data is properly sampled.
        - We should use best-case values (contamination delay) for calculating hold time slack
        - $T_"arr" = T_"ccq" + T_"cd"$
        - $T_"req" = T_"hold"$
        - $"hold slack" = T_"arr" - T_"req"$

    - *Clock Skew* is real, and we need to account for this when calculating setup and hold slack.
        - $t_"skew"$ gets accounted for in $T_"req"$ when calculating hold and setup slack.
        - *Clock Skew* can be useful--borrowing/retiming. 

#pagebreak()

== Datapath Elements, Memory
    - Non-Volatile Memory means data persists even when power is disconnected.
    - RAM - read/write memory, and you can access addresses randomly.
        - Dynamic: capacitors
        - Static: bi-stable latches
    - ROM - read-only memory
    - CAM - access by value

    === SRAM
        - Dense, higher area/capacity than FFs, not as dense as DRAM
        - Compatible with standard CMOS
        - Slower than FFs, but way faster than DRAM
        - *6T* vs *12T* SRAM
            - *6T* is more common, denser, used for large SRAM structures. Single word line for controlling reads/writes, differential pair for outputs
            - *12T* is used for smaller SRAMs, complementary read and write signals, tristate output for reads to bit line.
        - requires precharging before read/write operations.
        - large SRAM structures are sub-optimal, so banks and sub-arrays are often used.

    === DRAM
        - Different fabrication process than CMOS
        - Standard DRAM cell is 1T, and uses a *trench capacitor* to store charge.
        - Organized as 2D arrays, often broke up into banks and sub-arrays
        - *Three Basic Operations*
            - *ACT* - activate (open) a row.
            - *CAS* - perform the read/write on an opened row.
            - *PRE* - close the row and pre-charge.
    *Row Policies*
    - *Open*: Assumes next DRAM request will be in the same row, keeps data in row buffer.
        - Best Case: next DRAM request reads from same row, don't need to interact with DRAM bank
        - Worst Case: next DRAM request is to another row, need to do PRE, ACT, and CAS
    - *Closed*: Assumes next DRAM request will be in another row--writes data from row buffer back to the bank.
        - Alleviates worst case: regardless of the next request, you just need to do ACT and CAS

    - *DRAM Organization (smallest-to-largest)* - Subarray, Bank, Rank, DIMM, Channel
    - Memory controllers determine how to service external read/write requests.
        - FRFCFS, FCFS, perhaps consider request priority...
        - Memory Command Queue (*MCQ*)
    - *HBM* - 3D stacked memory dies, connected via *TSVs*
        - much higher bandwidth than typical DDR memory

    === Other Memory Organizations
        - *Shift Registers* - covered them in ECE120, good for parallel-to-serial, serial-to-parallel conversion
            - must be careful about hold times.
        - *CAMs* - not the bar, used for fully-associative structures (TLBs, Out-of-Order Issue Queues, etc.)
            - 10T CAM cell, similar-ish structure to SRAM cell.
        - *ROM* - can be implemented with single transistor cells, hardwired to VDD or GND.
        - *Programmable ROMs* - uses floating-gate nMOS transistor
            - FN tunneling allows charges to tunnel through gate oxide, thus turning the nMOS off.
            - Can be erasable through UV light exposure, or electrically as with EEPROMs

#pagebreak()

== Static and Dynamic Power

*Dynamic Power*
    - Capacitor *consumes* energy when its charging (rising edge)
    - The energy of the capacitor is $U = C V^2$, but half of this is dissipated through the pMOS on rise.
    - $P = E / t = E f $ where $f$ is your switching frequency. 
    - Thus, we arrive at the equation: $P_"dyn" = n f C V^2$
        - Dynamic Power depends on the activity factor $alpha$ of the circuit--certain gates switch more than others.
        - Estimating Activity Factor: Let $P = "Prob"("node" i = 1)$, and $P' = 1 - P$
            - $alpha approx P P'$
        - ex: for a 2-input NAND, $P = 3/4$, thus $alpha = P P' = (3/4) * (1/4) = 3/16$
        - For large designs, we can generally say $alpha approx 0.1$
    - *Clock Gating* - turn off clock (reducing dynamic power considerably) in unused blocks.
        - reduces $alpha = 0$
        - adds extra logic in determining when we can safely turn off these blocks.
    - *Voltage and Frequency Scaling*
        - we can run our logic at the lowest possible voltage and Frequency s.t. it meets performance requirements.
        - separate power supplies for different blocks on the chip, level conversion when crossing voltage domains.
        - this can be done dynamically, as with *Dynamic Voltage and Frequency Scaling (DVFS)*
    
*Wires and Interconnects*
    - For modern chips, wire delay, power consumption, and reliability are important issues.
        - Smaller wires means lower capacitance, but higher resistance (and thus power draw)
        - Thus in layout, we want to use higher metal layers for high-fanout nets, clk, VDD, VSS
            - Leave lower metal layers for high-density cells and short-distance routing.
    - Die-to-Die interconnections
        - Chiplets (AMD used SerDes to connect multiple dies together, but they now use straight wires I believe)
        - 3D stacking
    - Until 180nm generation, most wires were aluminum.
        - Modern processes use Cu, Co, Ru.
            - Cu atoms can diffuse into silicon and thus damage FETs, so we need a diffusion barrier to prevent this.
    - We can use repeaters to ensure signal integrity.

    - Need accurate modeling & aggressive optimizations.

= Final 

== Reliability, DFT 
    - Variation Sources: *PVT*
        - Process: process variation (at manufacturing)
        - Voltage: voltage fluctuation during operation
        - Temperature: operating temperature (environmental)
    - Designers need to account for these when modeling their chip.
    