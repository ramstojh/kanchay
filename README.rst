
*kanchay* means light in the Inca–Andean–Quechua cosmovision. *kanchay* is a simple tool based on `lightkurve <https://docs.lightkurve.org/>`_, `starspot <https://starspot.readthedocs.io/en/latest/index.html#/>`_ and `exoplanet <https://docs.exoplanet.codes/en/stable//>`_ codes to download light curves and measure their rotational periods using methods such as Lomb-Scargle Periodograms, autocorrelation functions (ACFs), Phase Dispersion Minimization (PDM), and Gaussian Processes (GPs). Such methods are well described in `starspot <https://starspot.readthedocs.io/en/latest/index.html#/>`_ and `exoplanet <https://docs.exoplanet.codes/en/stable//>`_.

Installation
------------
Installing *kanchay* requires only one step. Please run the follwing pip command::

    pip install kanchay

Dependencies
------------
The mean dependences of *kanchay* are  `lightkurve <https://docs.lightkurve.org/>`_, `starspot <https://starspot.readthedocs.io/en/latest/index.html#/>`_ and `exoplanet <https://docs.exoplanet.codes/en/stable//>`_. For installing these codes, please see their installation instructions. The otherd dependences are installed using pip::

    pip install matplotlib tqdm numpy pandas pymc3 Theano exoplanet
    
Example usage
-------------

.. code-block:: python

    from kanchay import kan
    
    # Search TESS light curve of a given star
    # The tool download and plot all the SPOC light curves (LC) observed by TESS
    # The tool also normalize and apply a sigma clipping to the LCs. The normalized LC is stores in x (time), y (flux) an yerr (flux error).
    starid='Gaia DR2 2395082091638750848'
    x, y, yerr = kan.search_lk(starid, mission='TESS')
    
    #you can plot a region of the LC
    import matplotlib.pyplot as plt
    plt.scatter(x[0][2000:8500], y[0][2000:8500])
    
    #
    kan.rotation_calc(x[0][2000:8500], y[0][2000:8500], yerr[0][2000:8500])
    
    #
    kan.rotation_calc(x[0][2000:8500], y[0][2000:8500], yerr[0][2000:8500], gp='yes')
    

Contributing
------------
*kanchay* is a simple tool created to help undergraduate studends to measure stellar rotational periods in a easy way, therefore the tool needs input improve. Please contact me (ramstojh@alumni.usp.br) if you have questions. Users are welcome propose new features, or report a bugs by opening an issue on GitHub.


Author
------
- `Jhon Yana Galarza <https://github.com/ramstojh>`_

License & attribution
---------------------

Copyright 2021, Jhon Yana Galarza.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite this package and its dependencies.
