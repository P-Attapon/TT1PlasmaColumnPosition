from .parameters import calibration_coeff

def magnetic_field_calibration(raw_B, kt, It, koh, Ioh, kv, Iv):
    """
    correct magnetic noise using calibration factors and machine's current
    :param raw_B: raw magnetic signal recorded from the magnetic probes
    :param kt: toroidal correction coefficient
    :param It: toroidal field current
    :param koh: ohmic correction coefficient
    :param Ioh: ohmic heating current
    :param kv: vertical field correction
    :param Iv: vertical field current
    :return: corrected magnetic field
    """
    return raw_B - (kt * It + koh * Ioh +  kv * Iv)

