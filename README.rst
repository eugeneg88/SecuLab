SecuLab - Solves secular equations of motion with various additional effects
==================================================

Welcome to SecuLab!

SecuLab is a Python package for the integration of the secular equations of motion of three-body hierarchical systems.
Written by Evgeni Grishin for multi-purpose code ffor study of various dynamical configurations focusing of hierarchical triples.

The options for secular equations include
---------------------------------------

**Secular evolution to octupole order** 

**Additoinal forth body**

**single-averaging corrections**

Based on `Luo et al. (2016) <http://adsabs.harvard.edu/abs/2016MNRAS.458.3060L>`_ and `Grishin et al. (2018) <http://adsabs.harvard.edu/abs/2018MNRAS.481.4907G>`_.

The options for additional forces include
-------------------------------------------

**Conservative short-range forces**
(GR, tides), based on `Liu et al. (2015) <http://adsabs.harvard.edu/abs/2015MNRAS.447..747L>`_.

**Tidal friction**
Equilibrium tides from Hut (1981) (e.g. Fabrycky and Tremane (2007) for more modern equations). In addition, a simplified version of dynamical tides is possible (e.g. Moe and Kratter, 2018), with some modifications.

**Gravitational wave emission and inspiral**
Compared to Peters (1964) formulae

**Simplified Galactic tides**
For wide binaries, based on Heisler and Tremanie (1986)

Clone / Download
--------

You need to have git installed. In addition, you need the NumPy and SciPy Python packages. 
Some tests require comparison with direct N-body code. I'm using  `REBOUND <https://rebound.readthedocs.io/en/latest/>`_, a high order accurate integrator.

.. code:: python
   
   git clone https://github.com/eugeneg88/seculab.git
   
Tests
===================

This is still under construction!
To test some of the features, I've created some scripts with test cases.

Quadrupole evolution 
------------------------

This test should reproduce Figure 4. of  `Naoz et al. (2013) <http://adsabs.harvard.edu/abs/2013MNRAS.431.2155N>`_ (Also Fig 3. in the `recent review <https://www.annualreviews.org/doi/10.1146/annurev-astro-081915-023315>`_ ).

.. code:: python
   
   import sl_tests as slt
   rebound_flag = True; t_end_myr = 1
   slt.test_quadupole_tpq(rebound_flag,t_end_myr)
 
After about 5 minutes of integration you should get something like this:

.. class:: no-web
	   
   .. image:: test_quadrupole_tpq.png
      :height: 100px
      :width: 200 px
      :scale: 100 %
You can turn off the n_body comparison by setting

.. code:: python

   rebound_flag = False
   
Which will speed up the integration. You can also contril the end time of he integration by changing t_end_myr.

Circumbinary planets 
------------------------

This script reproduces Fig. 3 of `Martin and Triaud (2016) <http://adsabs.harvard.edu/abs/2016MNRAS.455L..46M>`_:

.. code:: python
   
   import sl_tests as slt
   rebound_flag = True; t_end_myr = 1; single_averaging_flag = False;
   incs = [157, 158, 159]
   slt.circumbinary_planets(rebound_flag,t_end_myr, single_averaging_flag, incs)
   
It might take about an hour to integrate with REBOUND the ~ 10^7 orbits up to 1 Myr, but eventually you will see something like this

.. class:: no-web
	   
   .. image:: circumbinary_planets_sa_off.png
      :height: 100px
      :width: 200 px
      :scale: 100 %

The dashed lines are the exact N-body while the solid lines are SecuLab ibtegration. The results will fit a little better little better if we turn on the effective single averaging correction (more on that later!)

.. code:: python
	single_averaging_flag = True;

This will reproduce a similar plot, only the first spikes are captured slightly better. 

.. class:: no-web
	   
   .. image:: circumbinary_planets_sa_on.png
      :height: 100px
      :width: 200 px
      :scale: 100 %
      
Effective single averaging 
-------------------------------

It is possible to add an effective force / potential that mimics the secular evolution with corrections from short-term variations of the orbital elements. The corrections is based on `Luo et al. (2016) <http://adsabs.harvard.edu/abs/2016MNRAS.458.3060L>`_ and `Grishin et al. (2018) <http://adsabs.harvard.edu/abs/2018MNRAS.481.4907G>`_.

The following test reproduces fig. 1 of `Luo et al. (2016) <http://adsabs.harvard.edu/abs/2016MNRAS.458.3060L>`_: 

