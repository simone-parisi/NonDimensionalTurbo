import CoolProp as cp
import numpy as np
import matplotlib.pyplot as plt
from NonDimensionalTurbo import CentrifugalCompressor,RadialTurbine

if __name__=="__main__":
    fluidC=cp.AbstractState("HEOS","Air")
    compr=CentrifugalCompressor()
    compr.set_CoolProp_fluid(fluidC)
    compr.data_in["backsweep"]=30
    mass_flow=.1
    p_in=1e5
    T_in=300
    p_out=2e5
    compr.CoolProp_solve_outlet_fixed_P(mass_flow,p_in,T_in,p_out)
    #compr.CoolProp_solve_outlet_fixed_P(mass_flow,p_in,T_in,p_out,iterate_on_enthalpy=True) #gives same result, slower, might be more robust near two phase
    dataC=compr.get_output_data()
    print(dataC)
    phivec=np.linspace(0.01,0.15)
    etavec=np.zeros_like(phivec)
    RPMvec=np.zeros_like(phivec)
    for i,phi in enumerate(phivec):
        compr.data_in["flow_coefficient"]=phi
        compr.CoolProp_solve_outlet_fixed_P(mass_flow,p_in,T_in,p_out)
        etavec[i]=compr.get_output_data()['correlation_efficiency']*100
        RPMvec[i]=compr.get_output_data()['angular_speed']*30/np.pi
    plt.plot(RPMvec/1000,etavec,label="compressor (polytropic)")

    #example with mixture, does work only with REFPROP
    #fluidT=cp.AbstractState("REFPROP","N2&O2&CO2&H2O")
    #fluidT.set_mole_fractions([.8,.1,.05,.05])
    #fluidT.specify_phase(cp.iphase_gas)
    fluidT=fluidC
    turb=RadialTurbine()
    turb.set_CoolProp_fluid(fluidT)
    turb.data_in["clearance_height_ratio"]=0.04
    mass_flow=.1
    p_in=1.5e5
    T_in=800
    p_out=1.0e5
    turb.CoolProp_solve_outlet_fixed_P(mass_flow,p_in,T_in,p_out)
    #turb.CoolProp_solve_outlet_fixed_P(mass_flow,p_in,T_in,p_out,iterate_on_enthalpy=True) 
    dataT=turb.get_output_data()
    print(dataT)
    omsvec=np.linspace(0.3,0.9)
    etavec=np.zeros_like(omsvec)
    RPMvec=np.zeros_like(omsvec)
    for i,oms in enumerate(omsvec):
        turb.data_in["specific_speed"]=oms
        turb.CoolProp_solve_outlet_fixed_P(mass_flow,p_in,T_in,p_out)
        etavec[i]=turb.get_output_data()['correlation_efficiency']*100
        RPMvec[i]=turb.get_output_data()['angular_speed']*30/np.pi
    plt.plot(RPMvec/1000,etavec,label="turbine (isentropic)")
    plt.xlabel("kRPM")
    plt.ylabel("Efficiency [%]")

    plt.legend()
    plt.title("Comparison of compressor and turbine performance trend with RPM")
    plt.show()
    
        
