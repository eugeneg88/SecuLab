import seculab_main as sl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rebound

#logical flags, defaults
def test_isolated_gw_emission(e_iso_0, a1, mass1, mass2, t_end_myr, peters_flag):
    sl.octupole_flag = False
    sl.backreaction_flag = False
    sl.galactic_tide_flag = False
    sl.quad_flag = False
    sl.conservative_extra_forces_flag = True
    sl.GW_flag = True
    sl.single_averaging_flag = False
    sl.diss_tides_flag = False

    " initialize vectors with (e, inc, omega, Omega) - inc in RAD! "
    bin_A_vec = sl.init_binary(e_iso_0, 1e-8*np.pi/180., 0*np.pi/180., 0)
    bin_B_vec = sl.init_binary(1e-8, 0*np.pi/180, 0, 0*np.pi)
    masses = [mass1*sl.msun, mass2*sl.msun, 1e-8*sl.msun, 0*sl.msun]
    smas = [a1*sl.au, 10*sl.au, 1e10*sl.au]
    rs =  [1e-4 * sl.rsun, 1e-4*sl.rsun]
    ks = [0.014, 2*0.25]
    visc_ts = [2e4/365.25*sl.sec_in_yr, 2e4/365.5*sl.sec_in_yr]

    sl.a_stop=0
    sol, t = sl.run_sim(t_end_myr, bin_A_vec, bin_B_vec, masses, smas, rs, ks, visc_ts, 10000)
    t_sim, eccA, incA, nodesA, omegaA, eccB, incB, nodesB, omegaB, incAB, smaA, smaB, pericenterA = sl.get_element_solution(sol, t, smas)
    
#%%
## plotting ##
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(12,6))
    matplotlib.rcParams.update({'font.size': 24})
    plt.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.13, hspace = 0.22)

    plt.subplot(121)
    plt.plot([x/sl.sec_in_yr for x in t_sim], [x/sl.au for x in smaA], 'b', label = '$a$',  linewidth = 2);
    plt.plot([x/sl.sec_in_yr for x in t_sim], [x/sl.au for x in pericenterA], 'g', label = '$a(1-e)$',  linewidth = 2);
    plt.xlabel(r'$t\ [\rm yr]}$', fontsize=24); 
    plt.ylabel(r'$a\ [\rm AU]$', fontsize=24)
    plt.xscale('log');
    plt.yscale('log');
    plt.legend(loc = 'best')
    plt.ylim([2e-4, 1.1])

    plt.subplot(122)
    plt.plot([x/sl.sec_in_yr for x in t_sim],  np.ones(len(eccA)) - eccA , 'b', label = '$1-e_A$',  linewidth = 3);
    plt.ylabel('$1-e$', fontsize=24);
    plt.xscale('log')
    plt.yscale('log');
    plt.ylim([0.9997 - eccA[0], 1.02 - eccA[-1]])
    plt.xlabel(r'$t\ [\rm yr]}$', fontsize=24); 


    if peters_flag:
        c_0 = smas[0] * (1-e_iso_0**2)/e_iso_0**0.631578947368421/(1 + 121/304.*e_iso_0**2)**0.3784254023488473
        e_peters = np.logspace(-1, np.log10(e_iso_0),5000)
        a_peters = np.zeros(5000); a_peters2 = np.zeros(5000)
        for i in range(0,5000):
            a_peters[i] = c_0*e_peters[i]**0.631578947368421/(1-e_peters[i]**2)*(1 + 121/304.*e_peters[i]**2)**0.3784254023488473
        fig = plt.figure(2)
        plt.plot(np.ones(10000) - eccA,np.divide(smaA,sl.au),'b',label='sim', linewidth=3)
        plt.plot(np.ones(5000) - e_peters,a_peters/sl.au,'g--',label='Peters', linewidth=3)
        plt.xlabel('1-ecc'); plt.ylabel('a [AU]')
        plt.legend(loc=1)
        plt.xscale('log'); plt.yscale('log')
        plt.xlim([1-e_peters[-1], 1 - e_peters[0]])
        plt.subplots_adjust(left=0.2, right=0.94, top=0.93, bottom=0.13)

def test_quadupole_tpq(rebound_flag, t_end_myr):
    sl.octupole_flag = True
    sl.backreaction_flag = True
    sl.galactic_tide_flag = False
    sl.quad_flag = False
    sl.conservative_extra_forces_flag = False
    sl.GW_flag = False
    sl.single_averaging_flag = False
    sl.diss_tides_flag = False

    " initialize vectors with (e, inc, omega, Omega) - inc in RAD! "
    bin_A_vec = sl.init_binary(0.5, 70*np.pi/180., 120*np.pi/180., 0)
    bin_B_vec = sl.init_binary(0.00001, 0*np.pi/180, 0, 1*np.pi)
    masses = [1.4*sl.msun, 0.3*sl.msun, 0.01*sl.msun, 1e-16*sl.msun]
    smas = [5*sl.au, 50*sl.au, 1e10*sl.au]
    rs =  [1.0 * sl.rsun, sl.rsun]
    ks = [0.1, 0.25]
    visc_ts = [1*sl.sec_in_yr, 1*sl.sec_in_yr]

    sol, t = sl.run_sim(t_end_myr, bin_A_vec, bin_B_vec, masses, smas, rs, ks, visc_ts, 10000)
    t_sim, eccA, incA, nodesA, omegaA, eccB, incB, nodesB, omegaB, incAB, smaA, smaB, pericenterA = sl.get_element_solution(sol, t, smas)

    sl.backreaction_flag = False

    sol2, t2 = sl.run_sim(t_end_myr, bin_A_vec, bin_B_vec, masses, smas, rs, ks, visc_ts, 10000)
    t_sim2, eccA2, incA2, nodesA2, omegaA2, eccB2, incB2, nodesB2, omegaB2, incAB2, smaA2, smaB2, pericenterA2 = sl.get_element_solution(sol2, t2, smas)
    
    sl.backreaction_flag = True
    masses3 = [1.4*sl.msun, 1e-4*0.3*sl.msun, 0.01*sl.msun, 1e-16*sl.msun]
    sol3, t3 = sl.run_sim(t_end_myr, bin_A_vec, bin_B_vec, masses3, smas, rs, ks, visc_ts, 10000)
    t_sim3, eccA3, incA3, nodesA3, omegaA3, eccB3, incB3, nodesB3, omegaB3, incAB3, smaA3, smaB3, pericenterA3 = sl.get_element_solution(sol3, t3, smas)


#compare to N-body

    if rebound_flag:
        sim = rebound.Simulation()
        sim.add(m=1.4)
        sim.add(m=0.3, a=5., e=0.5, inc = 70*np.pi/180., omega=120*np.pi/180., Omega=0*np.pi/4., f=np.pi/4.)
        sim.add(m=0.01, a=50, e=0.00001, inc = 0*np.pi/180.0, omega=0, Omega=1*np.pi, f=0*np.pi/4. )
        sim.integrator = "ias15"
        sim.move_to_com() ##accounts for barycenter drift

        sim2 = rebound.Simulation()
        sim2.add(m=1.4)
        sim2.add(m=1e-4*0.3, a=5., e=0.5, inc = 70*np.pi/180., omega=120*np.pi/180., Omega=0*np.pi/4., f=np.pi/4.)
        sim2.add(m=0.01, a=50, e=0.00001, inc = 0*np.pi/180.0, omega=0, Omega=1*np.pi, f=0*np.pi/4. )
        sim2.integrator = "ias15"
        sim2.move_to_com() ##accounts for barycenter drift
        
        Noutputs = 2000
        times = np.linspace(0, t_end_myr*2*np.pi*1e6, Noutputs)
        inc1 = np.zeros(Noutputs); inc1tpq = np.zeros(Noutputs);  
        inc2 = np.zeros(Noutputs); inc2tpq = np.zeros(Noutputs);  
        inctot = np.zeros(Noutputs); inctottpq = np.zeros(Noutputs);  
        omega1 = np.zeros(Noutputs); omega1tpq = np.zeros(Noutputs);
        e1 = np.zeros(Noutputs);  e1tpq = np.zeros(Noutputs);


        for i,timesim in enumerate(times):
            sim.integrate(timesim, exact_finish_time=0)
            inc1[i] = sim.particles[1].inc
            inc2[i] = sim.particles[2].inc
            e1[i] = sim.particles[1].e
            omega1[i] = sim.particles[1].omega
            inctot[i] =  np.arccos(np.cos(inc1[i])*np.cos( inc2[i]) + np.sin(inc1[i])*np.sin( inc2[i])*np.cos(sim.particles[1].Omega - sim.particles[2].Omega))      

            sim2.integrate(timesim, exact_finish_time=0)
            inc1tpq[i] = sim2.particles[1].inc
            inc2tpq[i] = sim2.particles[2].inc
            e1tpq[i] = sim2.particles[1].e
            omega1tpq[i] = sim2.particles[1].omega
            inctottpq[i] =  np.arccos(np.cos(inc1tpq[i])*np.cos( inc2tpq[i]) + np.sin(inc1tpq[i])*np.sin( inc2tpq[i])*np.cos(sim2.particles[1].Omega - sim2.particles[2].Omega))    
#%%
## plotting ##
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(14,10))
    matplotlib.rcParams.update({'font.size': 24})
    plt.subplots_adjust(left=0.16, right=0.96, top=0.93, bottom=0.1)

    plt.subplot(221)
    plt.suptitle('$a_1 =$' + str(smas[0]/sl.au) + '$\  a_2 = $' + str(smas[1]/sl.au), fontsize=32)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim],  eccA, 'r', alpha=0.5, linewidth = 2);
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim2],  eccA2, 'b',alpha=0.5,  linewidth = 2);
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim3],  eccA3, 'g',alpha=0.8,  linewidth = 2);

    if rebound_flag:
        plt.plot(times/1e6/2./np.pi, e1, 'r--', linewidth = 3)
        plt.plot(times/1e6/2./np.pi, e1tpq, 'b--', linewidth = 3)
    plt.ylabel('$e$', fontsize=32);

    plt.subplot(222)
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim], np.degrees(incAB), 'r', alpha=0.5, linewidth = 2);
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim2], np.degrees(incAB2), 'b',alpha=0.5,  linewidth = 2);
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim3], np.degrees(incAB3), 'g',alpha=0.8,  linewidth = 2);
    if rebound_flag:
        plt.plot(times/1e6/2./np.pi, np.degrees(inctot), 'r--', linewidth = 3)
        plt.plot(times/1e6/2./np.pi, np.degrees(inctottpq), 'b--', linewidth = 3)
    plt.xlabel(r'$t\ [\rm Myr]}$', fontsize=32); 
    plt.ylabel('$i$ ' '[deg]', fontsize=32)

    ax = plt.subplot(223)
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim], omegaA,'r', label = 'seculab full, $m_1=0.3 M_{\odot}$', alpha=0.5,  linewidth = 2);
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim2], omegaA2,'b', label = 'seculab TPQ, $m_1=0.3 M_{\odot}$', alpha=0.5,  linewidth = 2);
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim3], omegaA3,'g',label = 'sebulab full, $m_1=3\cdot 10^{-5} M_{\odot}$', alpha=0.8,  linewidth = 2);
    if rebound_flag:
        plt.plot(times/1e6/2./np.pi, omega1, 'r--', label = 'rebound, $m_1=0.3 M_{\odot}$', linewidth = 3)
        plt.plot(times/1e6/2./np.pi, omega1tpq, 'b--',label = 'rebound, $m_1=3\cdot 10^{-5} M_{\odot}$', linewidth = 3)
    ax.legend(fontsize= 26, bbox_to_anchor=(2.25, 0.9))
    plt.xlabel(r'$t\ [\rm Myr]}$', fontsize=32); 
    plt.ylabel(r'$\omega$  [rad]', fontsize=32)
    
    
def test_circumbinary_planets(rebound_flag, t_end_myr, single_averaging_flag, incs):
    sl.octupole_flag = True
    sl.backreaction_flag = True
    sl.galactic_tide_flag = False
    sl.quad_flag = False
    sl.conservative_extra_forces_flag = False
    sl.GW_flag = False
    sl.diss_tides_flag = False

    " initialize vectors with (e, inc, omega, Omega) - inc in RAD! "
    bin_A_vec = sl.init_binary(0.8e-4, incs[0]*np.pi/180., 0*np.pi/180., 0)
    bin_B_vec = sl.init_binary(0.8e-4, 0*np.pi/180, 0, 0*np.pi)
    masses = [1*sl.msun, 0.5*sl.msun, 0.05*sl.msun, 1e-16*sl.msun]
    smas = [0.5*sl.au, 5*sl.au, 1e10*sl.au]
    rs =  [1e-4 * sl.rsun, 1e-4 * sl.rsun]
    ks = [0.1, 0.25]
    visc_ts = [50*sl.sec_in_yr, 50*sl.sec_in_yr]

    sol, t = sl.run_sim(t_end_myr, bin_A_vec, bin_B_vec, masses, smas, rs, ks, visc_ts, 10000)
    t_sim, eccA, incA, nodesA, omegaA, eccB, incB, nodesB, omegaB, incAB, smaA, smaB, pericenterA = sl.get_element_solution(sol, t, smas)

    bin_A_vec = sl.init_binary(0.8e-4, incs[1]*np.pi/180., 0*np.pi/180., 0)

    sol2, t2 = sl.run_sim(t_end_myr, bin_A_vec, bin_B_vec, masses, smas, rs, ks, visc_ts, 10000)
    t_sim2, eccA2, incA2, nodesA2, omegaA2, eccB2, incB2, nodesB2, omegaB2, incAB2, smaA2, smaB2, pericenterA2 = sl.get_element_solution(sol2, t2, smas)
    
    bin_A_vec = sl.init_binary(0.8e-4, incs[2]*np.pi/180., 0*np.pi/180., 0)

    sol3, t3 = sl.run_sim(t_end_myr, bin_A_vec, bin_B_vec, masses, smas, rs, ks, visc_ts, 10000)
    t_sim3, eccA3, incA3, nodesA3, omegaA3, eccB3, incB3, nodesB3, omegaB3, incAB3, smaA3, smaB3, pericenterA3 = sl.get_element_solution(sol3, t3, smas)


#compare to N-body
    if rebound_flag:
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=0.5, a=0.5, inc = incs[0]**np.pi/180.)
        sim.add(m=0.05, a=5)
        sim.integrator = "ias15"
        sim.move_to_com() ##accounts for barycenter drift

        sim2 = rebound.Simulation()
        sim2.add(m=1.)
        sim2.add(m=0.5, a=0.5, inc = incs[1]**np.pi/180)
        sim2.add(m=0.05, a=5)
        sim2.integrator = "ias15"
        sim2.move_to_com() ##accounts for barycenter drift
        
        sim3 = rebound.Simulation()
        sim3.add(m=1.)
        sim3.add(m=0.5, a=0.5, inc = incs[2]*np.pi/180. )
        sim3.add(m=0.05, a=5)
        sim3.integrator = "ias15"
        sim3.move_to_com() ##accounts for barycenter drift
        
        Noutputs = 10000
        times = np.linspace(0, t_end_myr*2*np.pi*1e6, Noutputs)
        inc1 = np.zeros(Noutputs); inc1tpq = np.zeros(Noutputs);  inc1_3 = np.zeros(Noutputs);  
        inc2 = np.zeros(Noutputs); inc2tpq = np.zeros(Noutputs);  inc2_3 = np.zeros(Noutputs)
        inctot = np.zeros(Noutputs); inctottpq = np.zeros(Noutputs);  inctot3 = np.zeros(Noutputs)
        omega1 = np.zeros(Noutputs); omega1tpq = np.zeros(Noutputs); omega3 = np.zeros(Noutputs)
        e1 = np.zeros(Noutputs);  e1tpq = np.zeros(Noutputs); e1_3 = np.zeros(Noutputs)
        e2 = np.zeros(Noutputs);  e2tpq = np.zeros(Noutputs); e2_3 = np.zeros(Noutputs)


        for i,timesim in enumerate(times):
            sim.integrate(timesim, exact_finish_time=0)
            inc1[i] = sim.particles[1].inc
            inc2[i] = sim.particles[2].inc
            e1[i] = sim.particles[1].e
            e2[i] = sim.particles[2].e
            omega1[i] = sim.particles[1].omega
            inctot[i] =  np.arccos(np.cos(inc1[i])*np.cos( inc2[i]) + np.sin(inc1[i])*np.sin( inc2[i])*np.cos(sim.particles[1].Omega - sim.particles[2].Omega))      

            sim2.integrate(timesim, exact_finish_time=0)
            inc1tpq[i] = sim2.particles[1].inc
            inc2tpq[i] = sim2.particles[2].inc
            e1tpq[i] = sim2.particles[1].e
            e2tpq[i] = sim2.particles[2].e
            omega1tpq[i] = sim2.particles[1].omega
            inctottpq[i] =  np.arccos(np.cos(inc1tpq[i])*np.cos( inc2tpq[i]) + np.sin(inc1tpq[i])*np.sin( inc2tpq[i])*np.cos(sim2.particles[1].Omega - sim2.particles[2].Omega))    

            sim3.integrate(timesim, exact_finish_time=0)
            inc1_3[i] = sim3.particles[1].inc
            inc2_3[i] = sim3.particles[2].inc
            e1_3[i] = sim3.particles[1].e
            e2_3[i] = sim3.particles[2].e            
            omega3[i] = sim3.particles[1].omega
            inctot3[i] =  np.arccos(np.cos(inc1_3[i])*np.cos( inc2_3[i]) + np.sin(inc1_3[i])*np.sin( inc2_3[i])*np.cos(sim3.particles[1].Omega - sim3.particles[2].Omega))      


#%%
## plotting ##

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(14,10))
    matplotlib.rcParams.update({'font.size': 24})
    plt.subplots_adjust(left=0.10, right=0.96, top=0.93, bottom=0.1, hspace = 0.15)

    plt.subplot(221)
    plt.suptitle('$a_1 =$' + str(smas[0]/sl.au) + '$\  a_2 = $' + str(smas[1]/sl.au), fontsize=32)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim],  eccA, 'r', alpha=0.9, linewidth = 2);
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim2],  eccA2, 'b',alpha=0.9,  linewidth = 2);
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim3],  eccA3, 'g',alpha=0.8,  linewidth = 2);

    if rebound_flag:
        plt.plot(times/1e6/2./np.pi, e1, 'r--', alpha= 0.4,linewidth = 3)
        plt.plot(times/1e6/2./np.pi, e1tpq, 'b--',alpha= 0.4, linewidth = 3)
        plt.plot(times/1e6/2./np.pi, e1_3, 'g--',alpha= 0.4, linewidth = 3)

    plt.ylabel('$e_1$', fontsize=32);

    plt.subplot(222)
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim], np.degrees(incAB), 'r', alpha=0.9, linewidth = 2);
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim2], np.degrees(incAB2), 'b',alpha=0.9,  linewidth = 2);
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim3], np.degrees(incAB3), 'g',alpha=0.8,  linewidth = 2);
    if rebound_flag:
        plt.plot(times/1e6/2./np.pi, np.degrees(inctot), 'r--',alpha= 0.4, linewidth = 3)
        plt.plot(times/1e6/2./np.pi, np.degrees(inctottpq), 'b--',alpha= 0.4, linewidth = 3)
        plt.plot(times/1e6/2./np.pi, np.degrees(inctot3), 'g--',alpha= 0.4, linewidth = 3)
    plt.xlabel(r'$t\ [\rm Myr]}$', fontsize=32); 
    plt.ylabel('$i$ ' '[deg]', fontsize=32)

    ax = plt.subplot(223)
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim], eccB,'r', label = r'$ \Delta I = $' + str(incs[0]) + ' [deg]', alpha=0.9,  linewidth = 2);
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim2],eccB2,'b',label = r'$ \Delta I = $' + str(incs[1]) + ' [deg]', alpha=0.9,  linewidth = 2);
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim3],eccB3,'g',label = r'$ \Delta I = $' + str(incs[2]) + ' [deg]', alpha=0.8,  linewidth = 2);
    if rebound_flag:
        plt.plot(times/1e6/2./np.pi, e2, 'r--', label = 'rebound', alpha= 0.4, linewidth = 3)
        plt.plot(times/1e6/2./np.pi, e2tpq, 'b--',alpha= 0.4, linewidth = 3)
        plt.plot(times/1e6/2./np.pi, e2_3, 'g--',alpha= 0.4, linewidth = 3)

    ax.legend(fontsize= 26, bbox_to_anchor=(2.0, 0.9))
    plt.xlabel(r'$t\ [\rm Myr]}$', fontsize=32); 
    plt.ylabel(r'$e_2', fontsize=32)
    
def test_single_averaging(rebound_flag, t_end_myr):
    sl.octupole_flag = True
    sl.backreaction_flag = True
    sl.galactic_tide_flag = False
    sl.quad_flag = False
    sl.conservative_extra_forces_flag = False
    sl.GW_flag = False
    sl.single_averaging_flag = False
    sl.diss_tides_flag = False
    
    " initialize vectors with (e, inc, omega, Omega) - inc in RAD! "
    bin_A_vec = sl.init_binary(0.2, 110*np.pi/180., 0*np.pi/180., 0)
    bin_B_vec = sl.init_binary(0.2, 0*np.pi/180, 0, 1*np.pi)
    masses = [1*sl.msun, 0.0001*sl.msun, 1*sl.msun, 1e-16*sl.msun]
    smas = [5*sl.au, 50*sl.au, 1e10*sl.au]
    rs =  [1.0 * sl.rsun, sl.rsun]
    ks = [0.1, 0.25]
    visc_ts = [1*sl.sec_in_yr, 1*sl.sec_in_yr]

    sol, t = sl.run_sim(t_end_myr, bin_A_vec, bin_B_vec, masses, smas, rs, ks, visc_ts, 10000)
    t_sim, eccA, incA, nodesA, omegaA, eccB, incB, nodesB, omegaB, incAB, smaA, smaB, pericenterA = sl.get_element_solution(sol, t, smas)

    sl.single_averaging_flag = True

    sol2, t2 = sl.run_sim(t_end_myr, bin_A_vec, bin_B_vec, masses, smas, rs, ks, visc_ts, 10000)
    t_sim2, eccA2, incA2, nodesA2, omegaA2, eccB2, incB2, nodesB2, omegaB2, incAB2, smaA2, smaB2, pericenterA2 = sl.get_element_solution(sol2, t2, smas)
#compare to N-body

    if rebound_flag:
        sim = rebound.Simulation()
        sim.add(m=1.)
        sim.add(m=0.0001, a=5., e=0.2, inc = 110*np.pi/180., omega=0*np.pi/180., Omega=0*np.pi/4., f=np.pi/4.)
        sim.add(m=1, a=50, e=0.2, inc = 0*np.pi/180.0, omega=0, Omega=1*np.pi, f=0*np.pi/4. )
        sim.integrator = "ias15"
        sim.move_to_com() ##accounts for barycenter drift
        
        Noutputs = 2000
        times = np.linspace(0, t_end_myr*2*np.pi*1e6, Noutputs)
        inc1 = np.zeros(Noutputs);
        inc2 = np.zeros(Noutputs); 
        inctot = np.zeros(Noutputs); 
        omega1 = np.zeros(Noutputs); 
        e1 = np.zeros(Noutputs);

        for i,timesim in enumerate(times):
            sim.integrate(timesim, exact_finish_time=0)
            inc1[i] = sim.particles[1].inc
            inc2[i] = sim.particles[2].inc
            e1[i] = sim.particles[1].e
            omega1[i] = sim.particles[1].omega
            inctot[i] =  np.arccos(np.cos(inc1[i])*np.cos( inc2[i]) + np.sin(inc1[i])*np.sin( inc2[i])*np.cos(sim.particles[1].Omega - sim.particles[2].Omega))      
## plotting ##
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(15,10))
    matplotlib.rcParams.update({'font.size': 24})
    plt.subplots_adjust(left=0.08, right=0.96, top=0.93, bottom=0.1, hspace=0.15)

    plt.subplot(311)
    plt.suptitle('$a_1 =$' + str(smas[0]/sl.au) + '$\  a_2 = $' + str(smas[1]/sl.au), fontsize=32)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim],  np.ones(len(eccA)) - eccA, 'r', label = 'DA only',  alpha=0.5, linewidth = 2);
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim2],  np.ones(len(eccB)) - eccA2, 'b',alpha=0.5, label = 'SA corrections',  linewidth = 2);

    if rebound_flag:
        plt.plot(times/1e6/2./np.pi, np.ones(len(e1)) - e1, 'g',  label = 'REBOUND', alpha=0.5, linewidth = 2)
    plt.ylabel('$1-e$', fontsize=32);
    plt.yscale('log')
    plt.legend(fontsize= 20, loc='best')


    plt.subplot(312)
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim], np.degrees(incAB), 'r', alpha=0.5, linewidth = 2);
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim2], np.degrees(incAB2), 'b',alpha=0.5,  linewidth = 2);
    if rebound_flag:
        plt.plot(times/1e6/2./np.pi, np.degrees(inctot), 'g',alpha=0.5, linewidth = 2)
    plt.xlabel(r'$t\ [\rm Myr]}$', fontsize=32); 
    plt.ylabel('$i$ ' '[deg]', fontsize=32)

    plt.subplot(313)
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim], omegaA,'r',alpha=0.5,  linewidth = 2);
    plt.plot([x/sl.sec_in_yr/1e6 for x in t_sim2], omegaA2,'b', alpha=0.5,  linewidth = 2);
    if rebound_flag:
        plt.plot(times/1e6/2./np.pi, omega1, 'g', alpha=0.5, linewidth = 2)
    plt.ylim([-3.2,3.2])
    plt.xlabel(r'$t\ [\rm Myr]}$', fontsize=32); 
    plt.ylabel(r'$\omega$  [rad]', fontsize=32)
 