SecuLab - Solves secular equations of motion with various additional effects
==================================================

Welcome to SecuLab!

SecuLab is a Python package for the integration of the secular equations of motion of three-body hierarchical systems.
Written by Evgeni Grishin for multi-purpose code ffor study of various dynamical configurations focusing of hierarchical triples.

The secular equations are up to octupole order, with option to add a forth body and single-averaging corrections (e.g. Luo et al. (2016) and Grishin et al. (2018))

Additional forces include:
- Conservative short-range forces (GR, tides), absed on Liu, Munoz and Lai (2015)
- Tidal friction: Equilibrium tides from Hut (1981) (e.g. Fabrycky and Tremane (2007) for more modern equations). In addition, a simplified version of dynamical tides is possible (e.g. Moe and Kratter, 2018), with some modifications.
- Gravitational wave emission and inspiral
- Simplified Galactic tides for wide binaries, based on Heisler and Tremanie (1986)

Installation
--------

You need to have git installed. In addition, you need the NumPy and SciPy Python packages.

.. code::
   
   git clone https://github.com/djmunoz/seculab.git

   cd seculab
   
   sudo python setup.py install

That is all!
 
