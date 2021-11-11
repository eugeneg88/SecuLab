#!/usr/bin/env python

""" 
Secuar Laboratory (SecuLab):
Solution to vector secular equations of a hierachical triples with perturbations          
                                                                                           
 			Author: Evgeni Grishin, September 2018	                                     
 			updates: Extended to 3 bodies and vertical Galactic tides, November 2018	  
                                                                                           
 Based on Hamers et. al (2015)[1], Liu, Munoz & Lai (2015)[2], Storch & Lai (2015)[3]  
 Solves the vector equations in octupole order with short-term corrections (Luo, Katz & Dong 2016)  
         
Added:                                                                                    
    Backreaction on binary B: Both in [2] and Petrovich (2015; ApJ 799:27)[4]  (Jan. 2019) 
    Consevative 1PN GR and equilibrium tides: Fabrycky & Tremaine (2007),[4] (Jan. 2019) 
    Dynamical tides:  Moe & Kratter (2018) [5] (requires additoinal parameters; Jan. 2019)  
    2.5PN terms and GW emission: Peters 1964                                               
    Single averaging: Grishin et al. 2018, private notes, Luo Katz and Dong (2016)         
    Galactic tides (only vertical), Heisler and Tremaine (1986) in vector form             
    Quadruple 3+1 systems
    Simple stellar evolution for m<8 msun, up to WD phase (added November 2021)
        -> includes exponential mass loss and variation of stellar parameters    

Stopping conditions allow checking for separation or pericentre limited by:
    1) Roche limit - tidal disruption - for q
    2) Schwartzshild radius - for q
    3) Roche Lobe overflow from Eggleton et al. (1998) - for q
    4) Dynamical instability following MArdling ans Aarseth (2001)
    5) User defined low q 'peri_stop'
    6) User defined low a 'a_stop'

Prerequirements:

All the standard python modules, numpy, scipy, ode solvers, matplotlib etc.

Some tests require REBOUND (Rein & Tamayo 2015) for comparison with direct N-body integrators
source code for the tests is ay seculab_tests.py
                                                                  
"""
# import modules for solving
from scipy.integrate import odeint
import numpy as np
from numpy import linalg as LA
import math

" Global constant and units: We work in cgs Gaussian units "
G = 6.674e-8 # Gravitational constant
msun = 2e33 # solar mass
rsun = 6.9e10 # solar radius
au = 1.5e13 # Astronomical unit
pc = 3.08567758e18 # parsec
c = 2.99792458e10 # speed of light
sec_in_yr = 3600*24*365.25 # seconds in year
mjup = 0.000954265748*msun #mass of Jupiter
" logical flags, defaults "

octupole_flag = True #Octupole evolution
backreaction_flag = True #Full (not test particle) limit
galactic_tide_flag = True #Galactic tides from Heisler and tremaine (1986)
quad_flag = False # quadruple 3+1 systems, not yet tested!
conservative_extra_forces_flag = True # GR, tidal and rotational bulges and extra precession
GW_flag = True # gravitational wave dissipation
single_averaging_flag = True # effective single averaging corrections
diss_tides_flag = True #dissipative tides
dynamical_variables_flag = True # for stellar evolution, adiabatic mass loss etc
#stopping conditions
stopping_roche_limit_flag = False
stopping_MA_stability_flag = True
stopping_rs_flag = True
stopping_rlo_flag = False
stopping_user_defined_flag = True

a_stop = 1 #cm # minimal separation
peri_stop = 1 # cm # minimal pericentre

" critical break up frequency"
def critical_spin(m,r):
    return np.sqrt(G * m / r**3)
    
" define gamma parameter for galactic tides or quadruples"
" change if required "
if galactic_tide_flag:
    gam = 0 
    rho_0 = 0.185 * msun / pc**3
if quad_flag:
    gam = 1
    m4 = msun
    a3 = 1e6*au
    e3=0.01

if diss_tides_flag:
    " dynamical tides efficiency parameter, [5]"
    f_dyn = 0.03
    " eccentricity below which dynamical tides are zero "
    e_dyn = 0.5
    """ 
    how exact is psudosyncronization: 
    exact ps is required only for extremely eccentric orbits
    """
    is_ps_exact = True
    if not is_ps_exact:
        expansion_order = 8
" Timescales "
" inner secular time " 
def t_sec_in(m1, m2, m3, a1, a2, e2):
    pin = 2.*np.pi / np.sqrt(G * binary_mass(m1,m2) / a1**3)
    pout = 2. * np.pi / np.sqrt(G * (m1+ m2 +m3) / a2**3)
    return (m1 + m2 + m3)  / m3 * pout**2 / pin * (1 - e2**2)**1.5 / 2. / np.pi

" outer secular time; either quadruple or galactic tide secular time "    
def t_sec_out(galactic_tide_flag, quad_flag, a2, a3,  m1, m2, m3, m4, e3):
    if not (galactic_tide_flag or quad_flag):
        return 1e300
    if galactic_tide_flag:
        return 3 /8. / np.pi / G / rho_0 * np.sqrt( G * (m1 + m2 + m3) / a2**3)
    if quad_flag:
        pin = 2.*np.pi / np.sqrt(G * (m1+m2+m3) / a2**3)
        pout = 2. * np.pi / np.sqrt(G * (m1+m2+m3+m4) / a3**3)        
        return (m1 + m2 + m3 + m4) / m4 * pout**2 / pin * (1 - e3**2)**1.5 / 2. / np.pi

" tidal friction time - for tidal dissipation - Eq. A9 of Fabrycky and Tremaine 2007"        
def tidal_fric_time (k1,r1, t_visc, m1, m2, a1):
    if k1 <= 1e-6: #allow defining k1=0 and avoid devergence
        return 1e300
    else:
        return t_visc/9.*(a1/r1)**8 * m1**2 / m2 / (m1+m2) / (1 + 2*k1)**2 
        
" conversion to orbital elements: h has the total angular momentum components "     
def get_orbital_elements(hx, hy, hz, ex, ey, ez):
    ret_ecc = (ex**2 + ey**2 + ez**2)**0.5
    ret_j = (1.0 - ret_ecc**2)**0.5
    total_ang_mom = (hx**2 + hy**2 + hz**2)**0.5
    
    jx = hx / total_ang_mom * ret_j;
    jy = hy / total_ang_mom * ret_j;
    jz = hz / total_ang_mom * ret_j;
    n = [-jy/(1.0-ret_ecc**2)**0.5, jx/(1.0-ret_ecc**2)**0.5, 0.0]
    ne_dot = (-jy*ex + jx*ey) / ret_ecc / (1.0-ret_ecc**2)**0.5

    ret_inc = np.arccos(jz/(1.0-ret_ecc**2)**0.5)
    ret_nodes = math.atan2(n[1],n[0]);
    ret_omega = math.atan2(ez, ne_dot * ret_ecc )
    
    return ret_ecc, ret_inc, ret_nodes, ret_omega, total_ang_mom
    
" binary and reduces masses "
def binary_mass (m1, m2):
    return m1 + m2

def reduced_mass (m1, m2):
    return m1*m2 / binary_mass(m1,m2)

"final WD masses - based on Catalan et al. (2008)"    
def wd_mass(m):
    if m<2.7*msun:
        wd_mass = 0.096*m + 0.429*msun
    else:
        wd_mass = 0.127*m + 0.318*msun
    return wd_mass

" circular canonical angular momenta "
def L_circ(m1,m2,a):
    return reduced_mass(m1,m2) * (G*binary_mass(m1,m2) * a)**0.5    
    
" viscous time and lag ime, footnote 1 of Grishin, Perets and Fragione (2018) "
def visc_to_tlag(k1,r,m,t_visc):
    return 1.5*(1 + 2*k1)**2/k1 * r**3 / G / m / t_visc

global time_series
time_series=-1 
""" MAIN EVOLUTION FUNCTION """
def evolve_binaries_quad (y, t, j_cx,j_cy,j_cz, m1, m2, m3, m4, a1, a2, a3,e3,  r1, k1, r2, k2, t_visc1, t_visc2):
     global state
     state = -1  # for stopping conditions  
     "Global variables to be returned if a stooping consition is satisfied"
     global a1_ret, a2_ret, e1_ret, e2_ret, iAB_ret, t_ret, iAB_ret, iBC_ret, iAC_ret, time_series

     " solution vector y = [e_A, j_A, e_B, j_B] "
     " The orbital elements and the energy, or SMA follow from e,j vectors alone "
     j_Ax, j_Ay, j_Az, e_Ax, e_Ay, e_Az, j_Bx, j_By, j_Bz, e_Bx, e_By, e_Bz = y

 
     """ 
     assignment of individual vectors 
	solution vectors - time varying
     Note that j is normalized to have e^2 + j^2 = 1, which is NOT the norm
     of the ang. mom. vector j_vec that has physical units!!!!!
      
     You should be extremely careful with the normalization, it cost me a lot of trouble!
     """
     djA_tot_dt = [0,0,0]; deA_tot_dt = [0,0,0]
     djB_tot_dt = [0,0,0]; deB_tot_dt = [0,0,0]
     
     e_A_vec = [e_Ax, e_Ay, e_Az]; e1 = LA.norm(e_A_vec); 
     j_A_vec = [j_Ax,j_Ay,j_Az]; j1 = (1-e1**2)**0.5;
     j_A_hat = [x/LA.norm(j_A_vec) for x in j_A_vec]
     e_B_vec = [e_Bx, e_By, e_Bz]; e2 = LA.norm(e_B_vec);
     j_B_vec = [j_Bx,j_By,j_Bz]; j2 = (1-e2**2)**0.5;
     j_B_hat = [x/LA.norm(j_B_vec) for x in j_B_vec]
     e_B_hat = [x/LA.norm(e_B_vec) for x in e_B_vec]    
     " z direction is binary B is embedded on the mid plane of the galaxy. "
     "usually in z direction, so j_cz = 1 with the rest zeros. "
     j_C_vec = [j_cx, j_cy, j_cz];
     j_C_hat = [x/LA.norm(j_C_vec) for x in j_C_vec]

     " some auxillary quantities "      
     e1sq = e1*e1
     e2sq = e2*e2
     j1sq  = 1 - e1sq
     j2sq = 1 - e2sq
              
     """ 
    Dynamical variables:
        Here you can vary the initial condition, e.g. mass radius etc. For example mass loss can change the mass
        radii may inflate and contract etc. 
        you need to turn on the dynamical_variables_flag and write your owm time dependent function for the evolution.
        need to check that the changeis sma is behaving properly, e.g.g add the mass changeinto angular momentum change etc.
     """ 
     if dynamical_variables_flag:
         """ A futile attampt for simple single stellar evolition of each of the inner binary stars"""
         global m1_init, m2_init, m3_init, m_wd_1, m_wd_2, a1_init, a2_init#, a1_actual_init, a2_actual_init
         global k1_init, k2_init, r1_init, r2_init,   t_visc1_init,   t_visc2_init
     # define timescales for RGB and WD transition
         t_ms1 = 1e10*sec_in_yr*(max(m1,m2)/msun)**-2.5
         t_ms2 = 1e10*sec_in_yr*(min(m1,m2)/msun)**-2.5
         t_wd1 = t_ms1*1.15
         t_wd2 = t_ms2*1.15

         if t <= 0.1*min(t_ms1,t_ms2):
    # at the MS save the initial parameters and fix them!
             m1_init = m1; m2_init = m2; m3_init=m3
             r1_init = r1; r2_init = r2;
             k1_init = k1; k2_init = k2;
             a1_init = a1; a2_init = a2; #a2_actual_init = a2_actual;

             t_visc1_init = t_visc1; t_visc2_init = t_visc2 
    #final WD masses from eq 2 and 3 of Catalan et al. (2008)
             m_wd_1 = wd_mass(m1_init)
             m_wd_2 = wd_mass(m2_init) 

         if t_wd1> t> t_ms1: #1st is RG, 2nd is MS
             m1_t = m1_init * (m1_init/m_wd_1)**((- (t-t_ms1) /  (t_wd1 - t_ms1)))
             r1 = 10 * r1_init
             k1 = 0.1
             t_visc1 = 5 * sec_in_yr * 0.014/k1
             a1 = a1_init * ((m1_t+m2_init)/(m1_init+m2_init))**-1
             a2 = a2_init * ((m1_t+m2_init + m3)/(m1_init+m2_init + m3))**-1
 
         if t_wd1 < t < t_ms2: #white dwarf 1, 2 still in ms
             m1 = m_wd_1
             a1 =  a1_init * ((m1+m2_init)/(m1_init+m2_init))**-1
             a2 = a2_init * ((m1+m2_init + m3)/(m1_init+m2_init + m3))**-1
             r1 = 0.01 * r1_init
             k1 = 0.1
             t_visc1 = 1e4 * sec_in_yr

         if t_wd2 > t > t_ms2: #1st is WD, 2nd is RG
             m1 = m_wd_1
             m2_t = m2_init * (m2_init/m_wd_2)**((- (t-t_ms2) /  (t_wd2 - t_ms2)))
             r2 = 10 * r2_init
             k2 = 0.1
             t_visc2 = sec_in_yr * 0.014/k2
             a1 = a1_init * ((m2_t+m_wd_1)/(m1_init+m2_init))**-1 
             a2 = a2_init * ((m2_t+m_wd_1 + m3)/(m1_init+m2_init +m3))**-1
          
         if t_wd2 < t : #both WDs
             m1 = m_wd_1
             m2 = m_wd_2
             a1 = a1_init * ((m1+m2)/(m1_init+m2_init))**-1
             a2 = a2_init * ((m1+m2+m3)/(m1_init+m2_init+m3))**-1

             r2 = 0.01 * r2_init
             k2 = 0.1
             t_visc2 = 1e4 * sec_in_yr
            
         peri_stop = 5*max(r1,r2) #update stopping condition for the pericentre

     " Semi-Major axes:  " 
     a1_actual = a1 * LA.norm(j_A_vec)**2/j1sq; 
     a2_actual = a2 * LA.norm(j_B_vec)**2/j2sq;
   
     """ stopping conditions """
     schwartzshild_radius = 2 * G * (m1+m2) / c**2
     roche_limit = max(r2 * (2 * m1 / m2)**(1/3.), r1 * (2 * m2 / m1)**(1/3.)) #wikipedia, find a better reference
     q = max(m1,m2) / min(m1,m2)     
     Eggleton_func = 0.49 * q**(2/3.) / (0.6 * q**(2/3.) + np.log(1 + q**(1/3.)))
     roche_overflow_radius = max(r1,r2)/Eggleton_func
     Y_MA = 2.8/(1 + e1) * ((1+m2/(m1+m2))*(1+e2)/(1-e2)**0.5 ) **(2/5.)
#     global state #decleration - preparing to change if needed
     """ 
     Here the stopping of the function is rather violent. 
     Oct 2020 update: Since I couldn't find a way to proprely stop, I just set the derivatives to zero.
     If anybody has a better solution to stop an odeint routine, let me know! 
     """
 #    if a1_actual * (1 - e1) <= max(peri_stop, roche_overflow_radius, 2.7* roche_limit, 3 * schwartzshild_radius) or a1_actual <= a_stop:
 
     if stopping_roche_limit_flag:         
         if a1_actual * (1 - e1) <= 2.7 * roche_limit:
 #            print('tidal radius reached! r_p/au, t/yr = ',a1_actual*(1-e1)/au, t/sec_in_yr)
             state = 1            
             return [0,0,0,0,0,0,0,0,0,0,0,0]
     if stopping_rs_flag:         
         if a1_actual * (1 - e1) <= 3 * schwartzshild_radius:
#             print('Schwartzshild radius reached! r_p/au, t/yr = ',a1_actual*(1-e1)/au, t/sec_in_yr)          
             state = 2
             return [0,0,0,0,0,0,0,0,0,0,0,0]
     if stopping_rlo_flag:         
         if a1_actual * (1 - e1) <= roche_overflow_radius:
#             print('Roche Lobe Overflow reached! r_p/au, t/yr = ',a1_actual*(1-e1)/au, t/sec_in_yr)          
             state = 3
             return [0,0,0,0,0,0,0,0,0,0,0,0]
     if stopping_MA_stability_flag:
         if a2_actual/a1_actual*(1 - e2) / (1 + e1) <= Y_MA:
#             print ('Triple is MA unstable! q2, ap1, Y, t/Gyr = ', a2_actual*(1-e2)/au, a1_actual*(1+e1)/au, Y_MA, t/sec_in_yr/1e9)
             state = 4
             if time_series <= 0:
                 time_series = t/sec_in_yr
             a1_ret = a1_actual/au 
             a2_ret = a2_actual/au
             e1_ret = e1
             e2_ret = e2
             iAB_ret = np.arccos(np.dot(j_A_hat, j_B_hat))
             iAC_ret = np.arccos(np.dot(j_A_hat, j_C_hat))
             iBC_ret = np.arccos(np.dot(j_B_hat, j_C_hat))
             return [0,0,0,0,0,0,0,0,0,0,0,0]
                
     if stopping_user_defined_flag:         
         if a1_actual * (1 - e1) <= peri_stop:
             state = 5
             if time_series <= 0:
                 time_series = t/sec_in_yr
             a1_ret = a1_actual/au 
             a2_ret = a2_actual/au
             e1_ret = e1
             e2_ret = e2
             iAB_ret = np.arccos(np.dot(j_A_hat, j_B_hat))
             iAC_ret = np.arccos(np.dot(j_A_hat, j_C_hat))
             iBC_ret = np.arccos(np.dot(j_B_hat, j_C_hat))
             return [0,0,0,0,0,0,0,0,0,0,0,0]
         
         if a1_actual  <= a_stop:
             state = 6
             if time_series <= 0:
                 time_series = t/sec_in_yr
             a1_ret = a1_actual/au 
             a2_ret = a2_actual/au
             e1_ret = e1
             e2_ret = e2
             iAB_ret = np.arccos(np.dot(j_A_hat, j_B_hat))
             iAC_ret = np.arccos(np.dot(j_A_hat, j_C_hat))
             iBC_ret = np.arccos(np.dot(j_B_hat, j_C_hat))
             return [0,0,0,0,0,0,0,0,0,0,0,0]
 #            print('Minimal semi major axis reached! a/au, t/yr = ',a1_actual/au, t/sec_in_yr)          

     " Timescales "
     " Here the factor 1/(1 - e2^2)^1.5 is already included, inner_time varies sine e2 varies "    
     inner_time = t_sec_in(m1, m2, m3, a1, a2, e2)
     outer_time = t_sec_out(galactic_tide_flag, quad_flag, a2, a3, m1, m2, m3, m4, e3)

     """ 
     Some additional auxillaries to be useful later.
     It is important to keep in mind that the dot with j is \hat{j} to avoid confusion.
     """
     " dot products: "
     dot_jAjB = np.dot(j_A_hat,j_B_hat)
     dot_eAjB = np.dot(e_A_vec,j_B_hat)
     dot_eAeB = np.dot(e_A_vec,e_B_hat)
     dot_jAeB = np.dot(j_A_hat,e_B_hat)
     dot_eAeB = np.dot(e_A_vec,e_B_hat)
     " cross products: "
     cr_jAjB = np.cross(j_A_hat,j_B_hat)
     cr_eAjB = np.cross(e_A_vec,j_B_hat)
     cr_jAeA = np.cross(j_A_hat,e_A_vec)
     cr_jAeB = np.cross(j_A_hat,e_B_hat)
     cr_eAeB = np.cross(e_A_vec,e_B_hat)
     
     " normalized angular momenta "
     ang_momHA  = LA.norm(j_A_vec)
     ang_momHB  = LA.norm(j_B_vec)
 
     " evolve binary A with time scale inner_time, Petrovich (2015) " 
     djA_dt1 = 3./4.*j1*dot_jAjB * cr_jAjB
     djA_dt2 = -15./4.* dot_eAjB * cr_eAjB/j1
      
     djA_tot_dt =+ ang_momHA * (djA_dt1 + djA_dt2)/inner_time 
 
     deA_dt1 = 3./4.*(dot_jAjB * cr_eAjB - 5 * dot_eAjB * cr_jAjB )
     deA_dt2 = 3./2.* cr_jAeA
      
     deA_tot_dt += (j1*deA_dt1 + j1*deA_dt2)/inner_time 
     
     " add octupole evolution "
     if octupole_flag:

         eps_oct = (m1 - m2) / (m1 + m2) * a1_actual/a2_actual * e2 / j2sq
         dot_jAjBsq = dot_jAjB * dot_jAjB
         dot_eAjBsq = dot_eAjB * dot_eAjB
         
         aux_term1 = (1.6 * e1*e1 - 0.2 - 7 * dot_eAjBsq + j1sq * dot_jAjBsq)
         aux_term2 =  dot_jAeB * dot_jAjB * j1sq - 7 * dot_eAeB * dot_eAjB 
         
         j_oct1 = 2 * ( dot_eAeB * dot_jAjB + dot_eAjB * dot_jAeB) * cr_jAjB * j1
         j_oct2 = 2 * aux_term2  * cr_eAjB / j1
         j_oct3 = 2 * dot_eAjB * dot_jAjB * cr_jAeB * j1
         j_oct4 =  aux_term1 * cr_eAeB / j1
         
         djA_tot_dt -= 1.171875 * eps_oct * ang_momHA *  (j_oct1 + j_oct2 + j_oct3 + j_oct4)/inner_time
         
         e_oct1 = 2 * dot_eAjB * dot_jAjB * cr_eAeB
         e_oct2 = aux_term1 * cr_jAeB
         e_oct3 = 2 * ( dot_eAeB * dot_jAjB + dot_eAjB * dot_jAeB ) * cr_eAjB
         e_oct4 = 2 * aux_term2  * cr_jAjB
         e_oct5 = 3.2 * dot_eAeB * cr_jAeA
         
         deA_tot_dt -= 1.171875 * eps_oct * j1 * (e_oct1 + e_oct2 + e_oct3 + e_oct4 + e_oct5)/inner_time

     " add conservative short range forces "
     if conservative_extra_forces_flag:
         " some auxillaries "
         a1_4th = a1_actual * a1_actual * a1_actual * a1_actual
         a2_3rd = a2_actual * a2_actual * a2_actual
         r1_5th = r1 * r1 * r1 * r1 * r1
         r2_5th = r2 * r2 * r2 * r2 * r2
         j1_10th = j1sq * j1sq * j1sq * j1sq * j1sq
         e1sq = e1 * e1
         e1_4th = e1sq * e1sq
         e1_8th = e1_4th * e1_4th
         e1_16th = e1_8th * e1_8th

         " From Liu, Munoz and Lai (2015) "
         eps_GR = 3.0 * G * (m1+m2) * (m1+m2) *a2_3rd* j2sq * j2 / a1_4th/c/c/m3
         deA_dt_GR = eps_GR*cr_jAeA/j1sq

         eps_tide1 = 15.0 * m1 * (m1 + m2) * a2_3rd * j2sq * j2 * k1 * r1_5th/a1_4th / a1_4th / m1 / m3
         eps_tide2 = 15.0 * m2 * (m1 + m2) * a2_3rd * j2sq * j2 * k2 * r2_5th/a1_4th / a1_4th / m2 / m3

         deA_dt_tide = (eps_tide1 + eps_tide2)*cr_jAeA/j1_10th * (1.0 + 1.5*e1sq + 0.125*e1_4th)
         
         deA_tot_dt += ( deA_dt_GR + deA_dt_tide)/inner_time 

     if diss_tides_flag:
         " equilibrium tidal model, Hut (1981), exact eqns. from Petrovich (2015) "
         " We assume pseudo-synchronization "
         
         " some auxillaries "
         a1_4th = a1_actual * a1_actual * a1_actual * a1_actual
         a2_3rd = a2_actual * a2_actual * a2_actual
         r1_5th = r1 * r1 * r1 * r1 * r1
         r2_5th = r2 * r2 * r2 * r2 * r2
         j1_10th = j1sq * j1sq * j1sq * j1sq * j1sq
         e1sq = e1 * e1
         e1_4th = e1sq * e1sq
         e1_8th = e1_4th * e1_4th
         e1_16th = e1_8th * e1_8th
         tidal_fric_time1  = tidal_fric_time (k1,r1, t_visc1, m1, m2, a1_actual)
         tidal_fric_time2  = tidal_fric_time (k2,r2, t_visc2, m2, m1, a1_actual)        
        
         f1v =  1 + 15/4.*e1sq + 15/8.*e1_4th + 5/64.*e1_4th * e1sq
         f2v =  1 + 3/2.*e1sq + 1/8.*e1_4th
         f1w =  1 + 15/2.*e1sq + 45/8.*e1_4th + 5/16.*e1_4th * e1sq
         f2w =  1 + 3*e1sq + 3/8.*e1_4th

         " pseudo-syncronization "  
         is_ps_exact = True

         if is_ps_exact:         
             ps =  f1w/f2w//j1sq/j1     
         else:
             cs = [6, 0.375, 173/8., - 4497/128., 4203/32., - 351329/1024., 1045671/1024., - 94718613/32768., 136576451/16384,-6258435315/262144., 17969605815/262144. ]
             es = [e1sq, e1_4th, e1_4th*e1sq, e1_8th, e1_8th*e1sq, e1_8th * e1_4th, e1_8th * e1_4th * e1sq, e1_16th, e1_16th * e1sq, e1_16th*e1_4th, e1_16th*e1_4th*e1sq]
             ps = 1
             global expansion_order
             if expansion_order < 6:
                 expansion_order = 6
             if expansion_order > 22:
                 expansion_order = 22
             loop_index = int(expansion_order/2)
             for i in range (0,loop_index):                
                 ps += cs[i] * es[i]

         V_term = 9 * (f1v / j1_10th / j1sq / j1 - 11/18.*ps*f2v / j1_10th ) 
         W_term = f1w / j1_10th / j1sq / j1 - ps*f2w/j1_10th 
         scalar_tot_V_term = V_term / tidal_fric_time1 + V_term / tidal_fric_time2
         scalar_tot_W_term =  W_term / tidal_fric_time1 + W_term / tidal_fric_time2 
        
         " equilibrium tides only " 
         if e1 < e_dyn: 
             V_dyn_term = 0
             " equilibriun and dynamical tides together "
             " simplified dynamical tides, Moe and Kratter (2018) "
         else:   
             dE_dyn1 = f_dyn * binary_mass(m1,m2)/m1 * G * m2**2 / r1 * (r1 / a1_actual / (1 - e1))**9
             dE_dyn2 = f_dyn * binary_mass(m1,m2)/m2 * G * m1**2 / r2 * (r2 / a1_actual / (1 - e1))**9
             E_orb = - G * m1 * m2 / 2 / a1_actual
             pin = 2.*np.pi / np.sqrt(G * binary_mass(m1,m2) / a1_actual**3)
             " eq 10 of Mor and Kratter 2018 " 
             da_dt_over_a =  (dE_dyn1 + dE_dyn2)/E_orb/pin
             " assuming conservation of angular momentum "
             V_dyn_term = - j1sq / 2. / e1 / e1 * da_dt_over_a
 
         V_final =  scalar_tot_V_term + V_dyn_term
         deA_tot_dt -= [V_final*x for x in e_A_vec]
         djA_tot_dt -= [scalar_tot_W_term*x for x in j_A_vec]
       
     " Add GW emission, 2.5PN terms "
     if GW_flag:
         scalar_e = 304/15.*G**3/c**5*m1*m2*(m1+m2)/a1_actual**4/j1**5*(1. + 121/304.*e1**2)
         scalar_h = 32/5.*G**3/c**5*(m1+m2)*m1*m2/a1_actual**4/j1**5*(1 + 7/8.*e1**2)
         
         deA_tot_dt -= [scalar_e*x for x in e_A_vec]
         djA_tot_dt -= [scalar_h*x for x in j_A_vec]
         
     " add signle averaging corrections  "
     if single_averaging_flag: # Eq 14 of Haim and Katz 2018 "
         eps_sa = (a1_actual/a2_actual)**1.5 * m3 / ((m1+m2) * (m1 + m2 + m3) )**0.5 / j2**3
         " define auxillaries if not already defined "         
         try: dot_jAjBsq
         except NameError: dot_jAjBsq = dot_jAjB * dot_jAjB 
         try: dot_eAjBsq
         except NameError: dot_eAjBsq = dot_eAjB * dot_eAjB         
         " circular terms: "
         Aj3 = 3 * (1 - 3 * j1sq * dot_jAjB**2 + 24*e1**2 - 15 * dot_eAjB**2)
         Ae3 = -90 * dot_jAjB * dot_eAjB * j1
         Aj6 = Ae3
         Ae6 = Aj3

         djA_tot_dt += 0.046875 * eps_sa * ang_momHA * (Aj3 * cr_jAjB + Aj6 * cr_eAjB/j1 ) / inner_time
         deA_tot_dt += 0.046875 * eps_sa * (144 * j1sq * dot_jAjB * cr_jAeA +  Ae3 * j1* cr_jAjB + Ae6 * cr_eAjB ) / inner_time

         " eccentric terms "
         if e2>=0.02:
            " outer unit vector triad v "
            v_B_hat = np.cross(j_B_hat,e_B_hat)
            " dot products "
            dot_eAvB = np.dot(e_A_vec, v_B_hat)
            dot_jAvB = np.dot(j_A_hat, v_B_hat)
            " cross_products "
            cr_eAvB = np.cross(e_A_vec, v_B_hat)
            cr_jAvB = np.cross(j_A_hat, v_B_hat)
            
            " define auxillaries if not already defined "         
            try: dot_jAeBsq
            except NameError: dot_jAeBsq = dot_jAeB * dot_jAeB         
            try: dot_jAvBsq
            except NameError: dot_jAvBsq = dot_jAvB * dot_jAvB         
            try: dot_eAeBsq
            except NameError: dot_eAeBsq = dot_eAeB * dot_eAeB         
            try: dot_eAvBsq
            except NameError: dot_eAvBsq = dot_eAvB * dot_eAvB         
         
            Aj1 =  10 * (dot_eAjB * dot_eAeB + j1sq * dot_jAjB * dot_jAeB )
            Ae1 =  10  * j1* (dot_eAjB * dot_jAeB + 13 * dot_jAjB * dot_eAeB )
            Aj2 = - 2 * (25 * dot_eAvB * dot_eAjB + j1sq * dot_jAjB * dot_jAvB )
            Ae2 = 10 *  j1* ( - 5 * dot_eAjB * dot_jAvB + 7 * dot_jAjB * dot_eAvB)
            Aj3_ecc = 5 *j1sq *  dot_jAeBsq - j1sq * dot_jAvBsq + 65 * dot_eAeBsq + 35* dot_eAvBsq
            Ae3_ecc = 10 *j1*  (dot_jAeB * dot_eAeB - 5 * dot_jAvB * dot_eAvB )
            Aj4 = Ae1
            Ae4 = Aj1
            Aj5 = Ae2
            Ae5 = Aj2
            Aj6_ecc = Ae3_ecc
            Ae6_ecc = Aj3_ecc
            
            djA_tot_dt += 0.046875 * eps_sa * e2**2 * ang_momHA * (Aj1 * cr_jAeB + Aj2 * cr_jAvB + Aj3_ecc * cr_jAjB + Aj4 * cr_eAeB/j1 + Aj5 * cr_eAvB/j1 + Aj6_ecc * cr_eAjB/j1) / inner_time
            deA_tot_dt += 0.046875 * eps_sa * e2**2 * (Ae1 *j1 * cr_jAeB + Ae2 *j1 * cr_jAvB + Ae3_ecc *j1 * cr_jAjB + Ae4 * cr_eAeB + Ae5 * cr_eAvB + Ae6_ecc * cr_eAjB) / inner_time
         
     " evolve binary B with time scale t_sec_out "
     if galactic_tide_flag or quad_flag:
         " define auxillaries if not already defined "         
         try: dot_jBjC
         except NameError: dot_jBjC = np.dot(j_B_hat,j_C_hat)    
         try: cr_jBjC
         except NameError: cr_jBjC = np.cross(j_B_hat,j_C_hat) 
         try: dot_eBjC
         except NameError: dot_eBjC = np.dot(e_B_vec,j_C_hat)    
         try: cr_eBjC
         except NameError: cr_eBjC = np.cross(e_B_vec,j_C_hat)
         try: cr_jBeB
         except NameError: cr_jBeB = np.cross(j_B_hat,e_B_vec)

         djB_dt1 = 3./4.*j2*dot_jBjC * cr_jBjC
         djB_dt2 = -15./4.*dot_eBjC * cr_eBjC/j2
   
         djB_tot_dt =  ang_momHB *(djB_dt1 + djB_dt2)/outer_time 
	
         deB_dt1 = 3./4.*(dot_jBjC * cr_eBjC - 5.*dot_eBjC * cr_jBjC)
         deB_dt2 = 3./4.*(1.+ gam)*cr_jBeB 
         
         deB_tot_dt+= (j2*deB_dt1 + j2*deB_dt2)/ outer_time

     " add back-reaction"
     if backreaction_flag:
         cr_jBeB = np.cross(j_B_hat,e_B_vec)
         L_ratio = L_circ(m1,m2, a1_actual) / L_circ(m1+m2 ,m3, a2_actual)
         
         djB_tot_dt -= L_ratio * ang_momHA * (djA_dt1 + djA_dt2)/inner_time 
         
         deB_br_dt1 = -3./4.* e2 * (j1sq * dot_jAjB * cr_jAeB  - 5 * dot_eAjB * cr_eAeB)
         deB_br_dt2 = -3/4.*(0.5 - 3*e1sq + 12.5*dot_eAjBsq - 2.5*j1sq*dot_jAjBsq)*cr_jBeB

         deB_tot_dt += L_ratio/j2*(deB_br_dt1 + deB_br_dt2)/inner_time
         
         if octupole_flag:
             djB_tot_dt += 1.171875 * L_ratio * eps_oct * ang_momHA *  (j_oct1 + j_oct2 + j_oct3 + j_oct4)/inner_time             
             
             term1 = 2 * (dot_eAjB * dot_jAeB + dot_jAjB * dot_eAeB) * np.cross(e_B_vec, j_A_vec)
             term2 = 2 * j2sq * dot_eAjB * dot_jAjB / e2 * np.cross(j_B_hat,j_A_vec)
             term3 = 2 * aux_term2  * np.cross(e_B_vec,e_A_vec)
             term4 = -j2sq/ e2 * aux_term1 * cr_eAjB # inverse cross product!!!!
             term5 =  ((0.4 - 3.2 * e1**2) * dot_eAjB + 14 * dot_eAjB * dot_jAeB * dot_jAjB + 7 * dot_eAeB * aux_term1 ) * cr_jBeB

             deB_tot_dt -= 1.171875 *L_ratio / j2 *  eps_oct * (term1 + term2 + term3 + term4 + term5) / inner_time
 
     """ END OF EVOLUTION EQUATIONS """
     
     " append evolution to solution vector" 	
     dydtA = np.append(djA_tot_dt, deA_tot_dt);
     dydtB = np.append(djB_tot_dt, deB_tot_dt);
     dydt = np.append(dydtA, dydtB) 
#  #   print (y0.shape)
     return dydt
     
" END OF EVOLUTION FUNCTION "
     
" ###### initial conditions ###### "
" initialalize binary vectoes from orbital elements "
def init_binary(eA_mag, incA0, omegaA0, nodesA0):
    jA_mag = (1.-eA_mag**2)**0.5
    jA_init = [jA_mag*np.sin(incA0)*np.sin(nodesA0), -jA_mag*np.sin(incA0)*np.cos(nodesA0), jA_mag*np.cos(incA0)]
    eA_init = [eA_mag*np.cos(omegaA0)*np.cos(nodesA0) - eA_mag*np.sin(omegaA0)*np.cos(incA0)*np.sin(nodesA0),eA_mag*np.cos(omegaA0)*np.sin(nodesA0) + eA_mag*np.sin(omegaA0)*np.cos(incA0)*np.cos(nodesA0),eA_mag*np.sin(omegaA0)*np.sin(incA0)]
    return np.append(jA_init, eA_init)

" run one simulation: "
def run_sim(t_end_myr, bin_A_vec, bin_B_vec, masses, smas, rs, ks, visc_ts, Nout):
    " construct solution vector "
    y0 = np.append(bin_A_vec, bin_B_vec)
#    print (y0.shape)    
    " constuct linear time series for solution "
    t = np.linspace(0,sec_in_yr*1e6*t_end_myr, Nout)    
    " unpack masses " 
    m1 = masses[0]
    m2 = masses[1]
    m3 = masses[2]
    m4 = masses[3]
    " unpack semi major axes"
    a1 = smas[0]
    a2 = smas[1]
    a3 = smas[2]
    " unpack radii "
    r1 = rs[0];
    r2 = rs[1]
    " unpack apsidal motion constants "
    k11 = ks[0];
    k12 = ks[1]
    " unpack viscous times  "
    t_visc1 = visc_ts[0]
    t_visc2 = visc_ts[1]

    " solve equations "
    global count
    count = 0 
    global mxstep
    mxstep=50000

    sol, infodict = odeint(evolve_binaries_quad, y0, t,  args= (0,0.0,0.1, m1, m2, m3, m4, a1, a2, a3, 0.01, r1, k11, r2, k12, t_visc1, t_visc2), mxstep = 50000, atol = 1e-12, rtol = 0.3e-12, full_output = True)
    return sol, t

def evolving_mass(m,t):
    t_ms = 1e10*sec_in_yr * (m/msun)**-2.5
    t_wd = t_ms*1.15
    m_wd = wd_mass(m)
    if t <= t_ms:
        mass_to_return = m
    elif t_ms <= t <= t_wd:
        mass_to_return  = m * (m/m_wd)**((- (t-t_ms) /  (t_wd - t_ms)))
    else: # t > t_wd:
        mass_to_return = m_wd
    return mass_to_return
    

" get orbital elements from solution vectors "
def get_element_solution(sol, t, smas, masses):
    nodesA = []; incA = []; incAB = []; eccA = []; omegaA= [];
    ang_momA = []; ang_momB =[];
    nodesB = []; incB = []; eccB = []; omegaB= [];
    smaA = []; smaB = []; pericenterA = []; t_sim = []
    oeA = []; oeB = []

    t_break=1e111
    for i in range(0, len(t)):
        oeA = get_orbital_elements(sol[i,0], sol[i,1], sol[i,2], sol[i,3] ,sol[i,4],  sol[i,5])
        oeB = get_orbital_elements(sol[i,6], sol[i,7], sol[i,8], sol[i,9] ,sol[i,10], sol[i,11])
        " check for NaNs and stop the function if NaNs detected "
        if math.isnan(oeA[0]):
            break     
        if t[i]/sec_in_yr > t_break:
            break
        # account for mass loss in the separations
        if dynamical_variables_flag:
            m1t = evolving_mass(masses[0], t[i])    
            m2t = evolving_mass(masses[1], t[i])    
            first_orbit_corr = (masses[0] + masses[1])/(m1t + m2t)            
            second_orbit_corr = (masses[0] + masses[1] + masses[2])/(m1t + m2t + masses[2])            

        t_sim.append(t[i])
        eccA.append(oeA[0]); eccB.append(oeB[0]);
        incA.append(oeA[1]); incB.append(oeB[1]); 
        nodesA.append(oeA[2]); nodesB.append(oeB[2]); 
        omegaA.append(oeA[3]); omegaB.append(oeB[3]);
        incAB.append (np.arccos(np.cos(incA[i])*np.cos( incB[i]) + np.sin(incA[i])*np.sin( incB[i])*np.cos(nodesA[i]-nodesB[i]))  )
        ang_momA.append(oeA[4])
        ang_momB.append(oeB[4])
        if dynamical_variables_flag:
            smaA.append(first_orbit_corr*smas[0]*ang_momA[i]*ang_momA[i]/(1. - eccA[i]*eccA[i]))
            smaB.append(second_orbit_corr*smas[1]*ang_momB[i]*ang_momB[i]/(1. - eccB[i]*eccB[i]))
        else:
            smaA.append(smas[0]*ang_momA[i]*ang_momA[i]/(1. - eccA[i]*eccA[i]))
            smaB.append(smas[1]*ang_momB[i]*ang_momB[i]/(1. - eccB[i]*eccB[i]))
        pericenterA.append(smaA[i] * (1 - eccA[i]))
    return t_sim, eccA, incA, nodesA, omegaA, eccB, incB, nodesB, omegaB, incAB, smaA, smaB, pericenterA


