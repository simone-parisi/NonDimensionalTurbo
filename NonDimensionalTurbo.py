# -*- coding: utf-8 -*-
"""
@author: sipar
NondimensionalRadial
A python module to estimate the efficiency of turbomachinery.
Based on the work presented in the paper:

Advancing non-dimensional models for radial turbomachinery ap-
plied to pumped thermal energy storage
Simone Parisi, Roberto Agromayor, Angelo La Seta,
Kenny Krogh Nielsen, Fredrik Haglind
In preparation (2025)

"""


import numpy as np
import scipy.optimize as sp
import CoolProp.CoolProp as cp
if __name__=="__main__":
    import matplotlib.pyplot as plt

NaN=float("NaN");

def friction_factor(Re,ksc):
    """
        
    """
    cfl=2.656/Re**0.5;
    cft=0.136/(-np.log10(0.2*ksc+12.5/Re))**2.15;
    t=5*(cfl/cft-1);
    P=1/(1+np.exp(-t));
    cf=P*cfl+(1-P)*cft;
    f=4*cf;
    return f

class _Machine:
    def set_CoolProp_fluid(self,fluid):
        self._fluid=fluid

    def _CoolProp_iterate_outlet_T(self,T_out):
        self._fluid.update(cp.PT_INPUTS,p_out,T_out)
        self.outlet={"T":self._fluid.T(),
                     "p":self._fluid.p(),
                     "hmass"  :self._fluid.hmass(),
                     "smass"  :self._fluid.smass(),
                     "rhomass"  :self._fluid.rhomass(),
                     "viscosity"  :self._fluid.viscosity(),
                     "speed_sound"  :self._fluid.speed_sound()}
        return self.residual()

    def CoolProp_solve_outlet_T(self,mass_flow,p_in,T_in,p_out):
        self.mass_flow=mass_flow
        self._fluid.update(cp.PT_INPUTS,p_in,T_in)
        self.inlet={"T":self._fluid.T(),
                    "p":self._fluid.p(),
                    "hmass"  :self._fluid.hmass(),
                    "smass"  :self._fluid.smass(),
                    "rhomass":self._fluid.rhomass(),
                    "viscosity" :self._fluid.viscosity(),
                    "speed_sound"  :self._fluid.speed_sound()}
        self._fluid.update(cp.PSmass_INPUTS,p_out,self.inlet["smass"])
        self.outlet_isentropic={"T":self._fluid.T(),
                     "p":self._fluid.p(),
                     "hmass"  :self._fluid.hmass(),
                     "smass"  :self._fluid.smass(),
                     "rhomass"  :self._fluid.rhomass(),
                     "viscosity"  :self._fluid.viscosity(),
                     "speed_sound"  :self._fluid.speed_sound()}
        #iterate with Brent method
        self._fluid.update(cp.PSmass_INPUTS,p_out,self.inlet["smass"])
        T_low=self._fluid.T()
        if p_out<p_in: #turbine -> real T is between inlet and isentropic
            T_high=self.inlet["T"]+(T_low-self.inlet["T"])*self.reference["K_loweff"]
        else: #compressor -> real T goes to inf as efficiency goes to 0
            T_high=self.inlet["T"]+(T_low-self.inlet["T"])/self.reference["K_loweff"]
        
        T_out=sp.brentq(self._CoolProp_iterate_outlet_T,T_low,T_high);
        residual=self._CoolProp_iterate_outlet_T(T_out)
        if residual >1e-9:
            print("Unable to find an outlet Temperature that satisfies the efficiency model")
        return T_out

    def get_output_data(self):
        return self._data.copy()

class CentrifugalCompressor(_Machine):
    def __init__(self):
        self.data_in={"flow_coefficient":0.07,
                      "backsweep":40.0,
                      "is_shrouded":False,
                      "has_vaned":True,
                      "roughness":3.6e-6,
                      "clearance":0.02,
                      "metal_density":7800,
                      "d_eta_other":0.0}
        self.reference={"diameter":0.4,
                        "peripheral_speed":277,
                        "roughness":3.6e-6,
                        "clearance":0.02,
                        "kinematic_viscosity":15.7e-6,
                        "K_loweff":0.5}
        self._fluid=None
        self.mass_flow = None
        self.inlet     ={"T":None,
                         "p":None,
                         "hmass"  :None,
                         "smass"  :None,
                         "rhomass":None,
                         "viscosity" :None,
                         "speed_sound"  :None}
        self.outlet    ={"T":None,
                         "p":None,
                         "hmass"  :None,
                         "smass"  :None,
                         "rhomass":None,
                         "viscosity" :None,
                         "speed_sound"  :None}
        self._data=None

    def residual(self):
        i=self.inlet
        o=self.outlet
        #compute polytropic head and efficiency
        dh=o["hmass"]-i["hmass"];
        dh_p=dh-(o["smass"]-i["smass"])*(o["T"]-i["T"])/np.log(o["T"]/i["T"]);
        eta_therm=dh_p/dh;
        #import primary parameters
        phi=self.data_in["flow_coefficient"]; #flow coefficient
        phx=4/np.pi*phi; #modified flow coefficient
        th=self.data_in["backsweep"];
        #baseline correlation
        if self.data_in["is_shrouded"]:
            I_ref=0.62-(phx/0.4)**3+0.0014/phx;
            mu_vnd=0.51+phx*(1-7.6*phx)-0.00025/phx;
            Kstress=0.4;
        else:
            I_ref=0.68-(phx/0.37)**3+0.002/phx;
            mu_vnd=0.59+phx*(0.7-7.5*phx)-0.00025/phx;
            Kstress=0.65+(0.75-0.67)/(0.12-0.5)*(max(0.05,phi)-0.05);
        #vaneless correction
        if self.data_in["has_vaned"]:
            eta_base=mu_vnd/I_ref;
        else:
            eta_vnd=mu_vnd/I_ref;
            eta_base=eta_vnd-0.017/(0.04+5*phx+eta_vnd**3);
        #work coefficient corrrection
        d_eta_th=-0.0006*(40-th);
        I=I_ref+(0.004+0.2*phi**2)*(40-th);
        
        #Peripheral speed, Diameter, Mach
        U=np.sqrt(dh/I);
        D=np.sqrt(self.mass_flow/(i["rhomass"]*phi*U));
        M=U/i["speed_sound"];
        
        #Mach correction
        P_m=max(0,phi*(M-0.8));
        d_eta_M=-(0.05+3*P_m)*P_m;
       
        #Local speed, chord, relative roughness
        w=U*(0.42+5*phi-14*phi**2);
        c=D/(1.1+22*phi+0.01*(40-th));
        ksc=(1*self.data_in["roughness"])/c;
        c_ref=c*(self.reference["diameter"]/D);
        ksc_ref=(1*self.reference["roughness"])/c_ref;                   
        
        # Reynolds number and friction factor
        kinematic_viscosity=i["viscosity"]/i["rhomass"]
        Re=w*c/kinematic_viscosity;
        Re_ref=Re*(self.reference["diameter"]/D)*(self.reference["peripheral_speed"]/U)/(self.reference["kinematic_viscosity"]/kinematic_viscosity);
        
        f    =friction_factor(Re    ,ksc    );
        f_ref=friction_factor(Re_ref,ksc_ref);
        # Reynolds number correction
        d_eta_Re=-(0.05+0.002/(phi+0.0025))*(f-f_ref)/f_ref;
        
        # Clearance correction
        d_eta_cl=-0.5*(self.data_in["clearance"]-self.reference["clearance"]);

        eta_corr=eta_base+d_eta_th+d_eta_M+d_eta_Re+d_eta_cl+self.data_in["d_eta_other"];
        
        residual=eta_corr-eta_therm;
        self._data     ={"correlation_efficiency":eta_corr,
                         "therm_efficiency"      :eta_therm,
                         "power"                 :dh*self.mass_flow,
                         "diameter"              :D,
                         "peripheral_speed"      :U,
                         "angular_speed"         :U/(D/2),
                         "mechanical_stress"     :Kstress*self.data_in["metal_density"]*U**2}
        
        return residual


class RadialTurbine(_Machine):
    def __init__(self):
        self.data_in={"specific_speed":0.6,
                      "is_scalloped":True,
                      "diffuser_loss_fraction":0.35,
                      "axial_clearance":0.02,
                      "radial_clearance":0.02,
                      "metal_density":7800,
                      "d_eta_other":0.0}
        self.reference={"axial_clearance":0.02,
                        "radial_clearance":0.02,
                        "Reynolds":352000.0,
                        "K_loweff":0.1}
        self._fluid=None
        self.mass_flow = None
        self.inlet     ={"T":None,
                         "p":None,
                         "hmass"  :None,
                         "smass"  :None,
                         "rhomass":None,
                         "viscosity" :None,
                         "speed_sound"  :None}
        self.outlet    ={"T":None,
                         "p":None,
                         "hmass"  :None,
                         "smass"  :None,
                         "rhomass":None,
                         "viscosity" :None,
                         "speed_sound"  :None}
        self.outlet_isentropic={"T":None,
                         "p":None,
                         "hmass"  :None,
                         "smass"  :None,
                         "rhomass":None,
                         "viscosity" :None,
                         "speed_sound"  :None}
        self.rotor_outlet_static={"T":None,
                         "p":None,
                         "hmass"  :None,
                         "smass"  :None,
                         "rhomass":None,
                         "viscosity" :None,
                         "speed_sound"  :None}
        self._data=None

    def compute_rotor_outlet_static(self,hmass,smass):
        self._fluid.update(cp.HmassSmass_INPUTS,hmass,smass)
        self.rotor_outlet_static={"T":self._fluid.T(),
                    "p":self._fluid.p(),
                    "hmass"  :self._fluid.hmass(),
                    "smass"  :self._fluid.smass(),
                    "rhomass":self._fluid.rhomass(),
                    "viscosity" :self._fluid.viscosity(),
                    "speed_sound"  :self._fluid.speed_sound()}
        return None
    
    def residual(self):
        i=self.inlet
        o=self.outlet
        #compute polytropic head and efficiency
        dh=o["hmass"]-i["hmass"];
        dh_is=self.outlet_isentropic["hmass"]-i["hmass"];
        eta_therm=dh/dh_is;
        #import primary parameters
        oms=self.data_in["specific_speed"];
        #baseline correlation
        eta_base=0.93-0.3*(oms-0.6)**2-0.5*(oms-0.6)**3
        if oms<0.45:
            eta_base-4.2*(oms-0.45)**2
        k_rot=max(0.02+0.08*oms**2,0.2*oms**4)
        #Clearance and scallop correction
        if self.data_in["is_scalloped"]:
            d_eta_scal=-0.03
        else:
            d_eta_scal=0.0
        d_eta_cl= -1.6*(self.data_in["radial_clearance"]-self.reference["radial_clearance"])
        d_eta_cl+=-0.15*(self.data_in["axial_clearance"]-self.reference["axial_clearance"])
        #overall efficiency
        eta_rot=eta_base+d_eta_cl+d_eta_scal+self.data_in["d_eta_other"];
        eta_overall=eta_rot/(1+self.data_in["diffuser_loss_fraction"]*k_rot);
        #Peripheral speed, Diameter, Mach
        nu=0.7
        dh_rot_is=dh_is/(1+self.data_in["diffuser_loss_fraction"]*k_rot);
        s_rot=i["smass"]+(o["smass"]-i["smass"])*(1/eta_rot-1)/(1/eta_overall-1);
        h_rot_s=o["hmass"]+0.5*k_rot*dh_rot_is;
        self.compute_rotor_outlet_static(h_rot_s,s_rot)
        U=nu*np.sqrt(2*(-dh_rot_is))
        om=oms*(-dh_rot_is)**0.75/(self.mass_flow/self.rotor_outlet_static["rhomass"])**0.5
        D=2*U/om
        Re=self.mass_flow/(i["viscosity"]*D/2);
        
        #Re correction
        eta_corr=1-(1-eta_overall)*(0.3+0.7*min(1,Re/self.reference["Reynolds"])**-0.2);

        #Mechanical stress
        Kstress=NaN
        
        residual=eta_corr-eta_therm;
        self._data     ={"correlation_efficiency":eta_corr,
                         "therm_efficiency"      :eta_therm,
                         "power"                 :-dh*self.mass_flow,
                         "diameter"              :D,
                         "peripheral_speed"      :U,
                         "angular_speed"         :om,
                         "mechanical_stress"     :Kstress*self.data_in["metal_density"]*U**2}
        
        return residual



if __name__=="__main__":
    fluidC=cp.AbstractState("HEOS","Air")
    compr=CentrifugalCompressor()
    compr.set_CoolProp_fluid(fluidC)
    compr.data_in["backsweep"]=30
    mass_flow=.1
    p_in=1e5
    T_in=300
    p_out=2e5
    compr.CoolProp_solve_outlet_T(mass_flow,p_in,T_in,p_out)
    dataC=compr.get_output_data()
    print(dataC)
    phivec=np.linspace(0.01,0.15)
    etavec=np.zeros_like(phivec)
    RPMvec=np.zeros_like(phivec)
    for i,phi in enumerate(phivec):
        compr.data_in["flow_coefficient"]=phi
        compr.CoolProp_solve_outlet_T(mass_flow,p_in,T_in,p_out)
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
    turb.data_in["axial_clearance"]=0.04
    turb.data_in["radial_clearance"]=0.04
    mass_flow=.1
    p_in=1.5e5
    T_in=800
    p_out=1.0e5
    turb.CoolProp_solve_outlet_T(mass_flow,p_in,T_in,p_out)
    dataT=turb.get_output_data()
    print(dataT)
    omsvec=np.linspace(0.3,0.9)
    etavec=np.zeros_like(omsvec)
    RPMvec=np.zeros_like(omsvec)
    for i,oms in enumerate(omsvec):
        turb.data_in["specific_speed"]=oms
        turb.CoolProp_solve_outlet_T(mass_flow,p_in,T_in,p_out)
        etavec[i]=turb.get_output_data()['correlation_efficiency']*100
        RPMvec[i]=turb.get_output_data()['angular_speed']*30/np.pi
    plt.plot(RPMvec/1000,etavec,label="turbine (isentropic)")
    plt.xlabel("kRPM")
    plt.ylabel("Efficiency [%]")

    plt.legend()
    plt.title("Comparison of compressor and turbine performance trend with RPM")
    plt.show()
    
        
