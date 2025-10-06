"""Geometry and dynamics for Lynx SES-PRO tubes and clamps."""

# Tube diameters (meters). SES-PRO tubes are Ø70 mm OD / Ø67 mm ID.
CF_TUBE_OD = 0.070
CF_TUBE_ID = 0.067
CF_TUBE_RAD = CF_TUBE_OD * 0.5

# Common catalog lengths (meters)
CF_LEN_118   = 0.1180
CF_LEN_143   = 0.1430
CF_LEN_280_5 = 0.2805
CF_LEN_305_5 = 0.3055

# Visual clamp ring length (meters)
CLAMP_RING_LEN = 0.034

# Mass split (tune to your needs)
STATOR_MASS = 0.02
ROTOR_MASS  = 0.04
TUBE_MASS   = 0.05

# End-effector flange + TCP defaults
EE_FLANGE_RADIUS = 0.035  # match tube OD (Ø70 mm -> r=35 mm)
EE_FLANGE_THICK  = 0.010
EE_MASS          = 0.10
TCP_OFFSET       = 0.060  # forward from flange face

# Small shrink to avoid z-fighting where cylinders meet
SHRINK = 0.995

# Servo defaults (affine position servo, as in robogen_lite style)
DEFAULT_KP = 1.0
DEFAULT_KV = 1.0
DEFAULT_CTRL_RANGE = (-1.5707963267948966, 1.5707963267948966)  # [-pi/2, +pi/2]