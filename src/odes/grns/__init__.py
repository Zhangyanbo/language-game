"""
Gene Regulatory Network (GRN) ODE Models
=========================================

Each .py file in this directory implements one GRN model from https://odebase.org.
All classes inherit from ODEBase (see ../ode_base.py).

Model Catalog
-------------
Circadian Clock:
  - Weimann2004  (7 sp)  — mammalian Per2/Bmal1, Hill + Michaelis-Menten
  - Almeida2019  (8 sp)  — E-box/R-box/D-box transcription mechanism
  - Leloup1999   (10 sp) — Drosophila PER/TIM, Hill n=4, phosphorylation cascade
  - Tyson1999*   (3 sp)  — minimal circadian with sqrt dimerization

Cell Cycle:
  - Gerard2010   (8 sp)  — mammalian Cyclin-CDK cascade, 2 constant species
  - Gardner1998  (5 sp)  — Goldbeter-Koshland ultrasensitive switches
  - Tyson1991*   (4 sp)  — minimal 2-var cell cycle (1 constraint)

Cell Fate Decision:
  - Chickarmane2006 (12 sp) — OCT4/SOX2/NANOG pluripotency network, 7 constant
  - Chickarmane2008 (6 sp)  — NANOG/GATA-6 bistable switch (reduced)

p53 / DNA Damage:
  - Zatorsky2006   (3 sp) — p53-MDM2 pulse oscillations

Synthetic Gene Circuits:
  - Liebal2012   (4 sp)  — B.subtilis transcription inhibition, 1 constant
  - Gardner2000  (3 sp)  — E.coli toggle switch, bistable, 1 constant

Signal Transduction:
  - Markevich2004* (5 sp) — MAPK double phosphorylation (1 constraint)
  - Kholodenko2000 (8 sp) — 3-tier MAPK cascade with MAPK-PP feedback (3 constraints)

(* = pre-existing models before this batch)


Workflow: Adding a New GRN Model
---------------------------------

1. **Choose a model on odebase.org**

   Browse https://odebase.org/table/ and pick a GRN.

   PITFALL: The odebase detail ID is NOT the BIOMD number. E.g.
   BIOMD0000000170 is at odebase detail ID 1432, not 170.  You must
   search the table page to find the correct detail ID.

2. **Gather information from sub-pages**

   For odebase detail ID <id>, the key sub-pages are:

   - Species map:      /detail/species_map/<id>/text
   - Parameter map:    /detail/parameter_map/<id>/text
   - ODEs:             /detail/odes/<id>/latex
   - Constraints:      /detail/constraints/<id>/latex
   - Parameter values:  /detail/parameters/<id>/latex

   PITFALL: The odebase detail page itself (e.g. /detail/1432/) sometimes
   returns 404, but the sub-pages still work.  Always try the sub-pages
   directly.

3. **Handle boundary / constant species**

   Many SBML models contain "boundary species" where dx/dt = 0. These
   are constants (gene loci, compartment volumes, fixed concentrations).
   Two approaches:

   (a) INCLUDE constants as species (Chickarmane2006): Keep NUM_NODES = 12
       even though only 5 are dynamic.  Set dx=0 for constant species.
       Pros: faithful to odebase formulation, easy to cross-reference.
       Cons: wastes reservoir dimensions on zeros, mu/std for constants
       need special treatment (set std >= 1.0 to avoid div-by-zero).

   (b) REDUCE to dynamic species only (Chickarmane2008): Set NUM_NODES = 6
       and store constants as self.k entries or instance attributes.
       Pros: every ODE dimension is useful for the reservoir.
       Cons: harder to map back to odebase x1..xN numbering.

   RECOMMENDATION: Approach (b) is preferred for reservoir computing.
   If the model has ≤2 constant species, approach (a) is acceptable.

4. **Handle assignment rules**

   Some parameters are not constants but computed from state variables.
   Example (Gardner1998): V1 = C * V1p / (C + K6).
   These MUST be computed inside ode(), not stored in self.k.

   PITFALL: Assignment rules appear in the parameter list on odebase
   but their "value" is an expression, not a number.  If you see a
   parameter with value 0 but the ODEs reference it, it's likely an
   assignment rule.

5. **Handle parameter aliases**

   Some models (Almeida2019) have "ModelValue" parameters that are just
   aliases for other parameters (e.g. k22 = k7 = V_D).  Odebase lists
   them as separate parameters.  Store them as self.k entries pointing
   to the same value, or compute shared sub-expressions once in ode().

6. **Create the .py file**

   Name: `authorYYYY_short_name.py`.  Class attributes:

     DESCRIPTION: str          NODE_NAMES: dict     PARAMETER_NAMES: dict
     URL: str                  NUM_NODES: int       NUM_PARAMS: int
     time_scale: float         mu: list[float]      std: list[float]

   Methods:
     __init__(self)           — self.k = [0] * (NUM_PARAMS + 1), 1-indexed
     ode(self, x)             — dx/dt, shape (..., NUM_NODES), differentiable
     project_to_manifold(x)   — identity if no constraints
     _random_initial_state()  — positive values in ORIGINAL (non-normalized) space

7. **Calibrate mu, std, time_scale**

   Write a test script that simulates ≥20 random initial states with RK4
   (dt=0.01, T=200, clamp min=1e-5).  Plot trajectories and compute stats.

   time_scale:
     - Oscillatory models: ≈ oscillation period
       (circadian ~24, cell cycle ~28-35, p53 ~8)
     - Bistable/steady-state models: ≈ settling time
       (toggle switch ~15, stem cell ~50-100)

   mu & std:
     - Compute from the LATTER HALF of trajectories to exclude transients
     - For constant species, use the fixed value as mu and std ≥ 1.0

   PITFALL: Species can span vastly different magnitudes.  E.g. in
   Liebal2012, sigb ~ 226,000 while lacz ~ 16.  Per-species normalization
   is critical.

8. **_random_initial_state: common pitfalls**

   - BISTABLE systems (Gardner2000, Chickarmane2006/2008): Uniform random
     sampling may only reach ONE basin.  Explicitly sample from both
     basins (see Gardner2000 for example).

   - CONSTANT species: Must be set to their fixed values, NOT randomized.
     Otherwise the ODE will try to evolve a "constant" from a wrong value.

   - RANGE too wide: If initial values are far from the attractor, RK4
     can diverge before clamp catches it.  Use observed oscillation
     ranges from calibration, not wild guesses.

   - Goldbeter-Koshland terms like V*(1-x)/(K+1-x): If x > 1+K, the
     denominator goes negative.  Constrain initial x ∈ [0, 1] for
     fraction-type species (e.g. active kinase fraction).

9. **Numerical stability in ode()**

   - Hill functions with large n (n=4, n=8): x^n can overflow for large x.
     The RK4 integrator's clamp(min=1e-5) prevents x from hitting exactly
     zero, which avoids 0^n = nan for fractional n.

   - Michaelis-Menten denominators V*x/(K+x): Always safe for x > 0, K > 0.
     Even for very small K (e.g. 0.0001 in Zatorsky2006), x > 0 ensures
     positivity.

   - Compartment volumes: Many models divide every ODE by a compartment
     volume = 1.  Keep the division for correctness even though it's a
     no-op, in case parameters are later changed.

   - Shared sub-expressions: For complex models (Almeida2019, Leloup1999),
     compute shared terms (Hill functions, complex formation rates) once
     and reuse them.  This improves both readability and performance.

10. **Register and validate**

    Add an import and __all__ entry below.  Then verify:

      m = NewModel()
      x = m.random_initial_state(4); x.requires_grad_(True)
      dx = m.ode(m.denormalize(x)); dx.sum().backward()
      assert x.grad is not None and not torch.isnan(x.grad).any()
      traj = m.simulate(m.random_initial_state(2), dt=0.01, T=0.1)
      assert not torch.isnan(traj).any()
"""

from .tyson1999_circle_lock import Tyson1999CircleLock
from .markevich2004_mapk import Markevich2004MAPKDoublePhosphorylation
from .tyson1991_cell_cycle import Tyson1991CellCycle2Var
from .weimann2004_circadian import Weimann2004CircadianOscillator
from .almeida2019_circadian import Almeida2019CircadianClock
from .zatorsky2006_p53 import Zatorsky2006P53Model4
from .gardner2000_toggle_switch import Gardner2000ToggleSwitch
from .liebal2012_transcription import Liebal2012TranscriptionInhibition
from .gerard2010_cell_cycle import Gerard2010CellCycle
from .chickarmane2006_stem_cell import Chickarmane2006StemCellSwitch
from .gardner1998_cell_cycle import Gardner1998CellCycleGoldbeter
from .leloup1999_circadian import Leloup1999CircadianClock
from .chickarmane2008_nanog_gata6 import Chickarmane2008NanogGata6
from .kholodenko2000_mapk_cascade import Kholodenko2000MAPKCascade

__all__ = [
    "Tyson1999CircleLock",
    "Markevich2004MAPKDoublePhosphorylation",
    "Tyson1991CellCycle2Var",
    "Weimann2004CircadianOscillator",
    "Almeida2019CircadianClock",
    "Zatorsky2006P53Model4",
    "Gardner2000ToggleSwitch",
    "Liebal2012TranscriptionInhibition",
    "Gerard2010CellCycle",
    "Chickarmane2006StemCellSwitch",
    "Gardner1998CellCycleGoldbeter",
    "Leloup1999CircadianClock",
    "Chickarmane2008NanogGata6",
    "Kholodenko2000MAPKCascade",
]
