
*kanchay* means light in the Inca–Andean–Quechua cosmovision. *kanchay* is a simple tool based on `lightkurve <https://docs.lightkurve.org/>`_, `starspot <https://starspot.readthedocs.io/en/latest/index.html#/>`_ and `exoplanet <https://docs.exoplanet.codes/en/stable//>`_ codes to download light curves and measure their rotational periods using methods such as Lomb-Scargle Periodograms (LS), autocorrelation functions (ACFs), Phase Dispersion Minimization (PDM), and Gaussian Processes (GPs). Such methods are well described in `starspot <https://starspot.readthedocs.io/en/latest/index.html#/>`_ and `exoplanet <https://docs.exoplanet.codes/en/stable//>`_.

Installation
------------
Installing *kanchay* requires only one step. Please run the following pip command::

    pip install kanchay

Dependencies
------------
The main dependencies of *kanchay* are  `lightkurve <https://docs.lightkurve.org/>`_, `starspot <https://starspot.readthedocs.io/en/latest/index.html#/>`_ and `exoplanet <https://docs.exoplanet.codes/en/stable//>`_. For installing these codes, please see their installation instructions. The other dependencies are installed using pip::

    pip install matplotlib tqdm numpy pandas pymc3 Theano exoplanet
    
Example usage
-------------

.. code-block:: python

    from kanchay import kan
    
    # Search TESS light curves of a given star
    # The tool downloads and plots all the SPOC light curves (LC) observed by TESS in several sectors
    # The tool also normalizes and applies sigma clipping to the LCs. The resultant LC is stored in arrays in x (time), y (flux) and yerr (flux error).
    starid='Gaia DR2 2395082091638750848'
    x, y, yerr = kan.search_lk(starid, mission='TESS')
    
    # You can also plot a region of interest of the LC
    import matplotlib.pyplot as plt
    plt.scatter(x[0][2000:8500], y[0][2000:8500])
    
    # You can determine rotational periods with only one command (thanks to the starspot code)
    # The rotational period is estimated using three methods (LS, ACF, PDM)
    kan.rotation_calc(x[0][2000:8500], y[0][2000:8500], yerr[0][2000:8500])
    
    # You can also determine rotational periods using GP (thanks to the exoplanet code)
    kan.rotation_calc(x[0][2000:8500], y[0][2000:8500], yerr[0][2000:8500], gp='yes')
    

Contributing
------------
*kanchay* is a simple tool created to help undergraduate students measure stellar rotational periods in an easy and simple way. Therefore, the tool needs input to improve. Please contact me (ramstojh@alumni.usp.br) if you have questions. Users are welcome to propose new features or report bugs by opening an issue on GitHub.


Authors
-------
- `Jhon Yana Galarza <https://github.com/ramstojh>`_
- `Adriana Valio <https://orcid.org/0000-0002-1671-8370>`_

License & attribution
---------------------

Copyright 2021, Jhon Yana Galarza.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite this package and its dependencies.
