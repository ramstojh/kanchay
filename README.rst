
*kanchay* means light in the Inca–Andean–Quechua cosmovision. *kanchay* is a simple tool based on lightkurve, starspot and exoplanet codes to download light curves and measure their rotational periods using methods such as Lomb-Scargle Periodograms, autocorrelation functions (ACFs), Phase Dispersion Minimization (PDM), and Gaussian Processes (GPs). Such methods are well described in `starspot <https://starspot.readthedocs.io/en/latest/index.html#/>`_ and `exoplanet <https://docs.exoplanet.codes/en/stable//>`_.


Example usage
-------------

.. code-block:: python

    from pachamama import terra
    
    # Computing convective mass of a star with [Fe/H] = 0.164 dex 
    # and mass = 1.14 solar masses
    # The mass can take values from 0.5 <= M <= 1.3 (solar mass)
    # The [Fe/H] can take values from -1.0 <= [Fe/H] <= 0.3 (dex)
    # By default the code computes the convective mass using the Yale isocrhones of stellar evolution
    terra.cvmass(feh=0.1, mass=1)
    
    
    # Computing the abundance pattern of a star with [Fe/H] = 0.164 dex, 
    # mass = 1.14 solar mass, Mcon (convective mass) 0.01. 
    # obs_abd.csv is a table containing the observed abundance. See the terra file example 
    terra.pacha(feh=0.164, mass=1, Mcon=0.01, data_input='obs_abd.csv')
    
    # If you want to save the outputs with a specific name (e.g., a star name).
    terra.pacha(feh=0.164, mass=1, Mcon=0.01, data_input='obs_abd.csv', data_output='HIP71726')


License & attribution
---------------------

Copyright 2021, Jhon Yana.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite this package and its dependencies.
