from dataclasses import dataclass

@dataclass
class SpaceMouseConfig:
    """
    Configuration class for a SpaceMouse device. It is NOT recommended to change these, with
    the exception of vendor_id and product_id, which should only be changed if using a different
    SpaceMouse device. To see the available devices, run hid.enumerate() in a Python interpreter

    :config_param pos_sensitivity: position sensitivity (do not change!)
    :config_param rot_sensitivity: rotation sensitivity (do not change!)
    :config_param verbose: adds helpful debugging print statements
    """
    pos_sensitivity: float = 1.0
    rot_sensitivity: float = 1.0
    verbose: bool = False
    vendor_id: int = 9583
    product_id: int = 50741

