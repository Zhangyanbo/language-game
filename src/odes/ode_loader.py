from . import (
    LorenzSystem,
    Tyson1999CircleLock,
    Markevich2004MAPKDoublePhosphorylation,
    Tyson1991CellCycle2Var,
    Weimann2004CircadianOscillator,
    Almeida2019CircadianClock,
    Zatorsky2006P53Model4,
    Gardner2000ToggleSwitch,
    Liebal2012TranscriptionInhibition,
    Gerard2010CellCycle,
    Chickarmane2006StemCellSwitch,
    Gardner1998CellCycleGoldbeter,
    Leloup1999CircadianClock,
    Chickarmane2008NanogGata6,
    Kholodenko2000MAPKCascade,
)

# Registry: short_name -> class
_ODE_REGISTRY = {
    "lorenz": LorenzSystem,
    "tyson1999": Tyson1999CircleLock,
    "markevich2004": Markevich2004MAPKDoublePhosphorylation,
    "tyson1991": Tyson1991CellCycle2Var,
    "weimann2004": Weimann2004CircadianOscillator,
    "almeida2019": Almeida2019CircadianClock,
    "zatorsky2006": Zatorsky2006P53Model4,
    "gardner2000": Gardner2000ToggleSwitch,
    "liebal2012": Liebal2012TranscriptionInhibition,
    "gerard2010": Gerard2010CellCycle,
    "chickarmane2006": Chickarmane2006StemCellSwitch,
    "gardner1998": Gardner1998CellCycleGoldbeter,
    "leloup1999": Leloup1999CircadianClock,
    "chickarmane2008": Chickarmane2008NanogGata6,
    "kholodenko2000": Kholodenko2000MAPKCascade,
}


def load_ode_model(ode_name: str):
    cls = _ODE_REGISTRY.get(ode_name)
    if cls is None:
        raise ValueError(
            f"Unknown ODE model: {ode_name}. "
            f"Expected one of: {', '.join(_ODE_REGISTRY.keys())}."
        )
    return cls()
