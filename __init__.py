__uri__ = "https://duste.readthedocs.io"
__author__ = "Gautam Nagaraj"
__email__ = "gxn75@psu.edu"
__license__ = "MIT"
__description__ = "Computation of dust attenuation curves for galaxies given stellar mass, star formation rate, stellar metallicity, redshift, and/or axis ratio"

from . import DustAttnCalc

__all__ = ["regular_grid_interp_scipy","mass_completeness","get_dust_attn_curve_d2","get_dust_attn_curve_d1","getMargSample","getTraceInfo","getModelSamplesI","plotDustAttn","plotDust12","DustAttnCalc"]