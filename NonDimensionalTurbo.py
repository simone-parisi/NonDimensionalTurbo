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

from math import nan,pi,log,exp,log10,sqrt
import scipy.optimize as sp
import CoolProp.CoolProp as cp

NaN=float("NaN");

def friction_factor(Re,ksc):
    """
        
    """
    # Laminar friction factor component
    cfl=2.656/Re**0.5;
    # Turbulent friction factor component 
    cft=0.136/(-log10(0.2*ksc+12.5/Re))**2.15;
    # Transition function 
    t=5*(cfl/cft-1);
    P=1/(1+exp(-t));
    cf=P*cfl+(1-P)*cft;
    # Darcy friction factor
    f=4*cf;
    return f

class _Machine:
    """Base class for turbomachinery models."""
    def set_CoolProp_fluid(self,fluid):
        """Sets the CoolProp fluid object for thermodynamic property calculations."""
        self._fluid=fluid
        self._iterate_on_enthalpy=False
    def _CoolProp_iterate_outlet_fixed_P(self,x_out):
        """
        Internal function called by the solver to iterate on outlet conditions.
        It calculates the residual between model-predicted efficiency and
        efficiency derived from thermodynamic states.
        """
        # Update outlet state based on current guess of x_out and known p_out
        if self._iterate_on_enthalpy:
            self._fluid.update(cp.HmassP_INPUTS,x_out,self.outlet_isentropic["p"])
        else:
            self._fluid.update(cp.PT_INPUTS,self.outlet_isentropic["p"],x_out)
        self.outlet={"T":self._fluid.T(),
                     "p":self._fluid.p(),
                     "hmass"  :self._fluid.hmass(),
                     "smass"  :self._fluid.smass(),
                     "rhomass"  :self._fluid.rhomass(),
                     "viscosity"  :self._fluid.viscosity(),
                     "speed_sound"  :self._fluid.speed_sound()}
        # Calculate and return the residual based on the machine's specific model
        return self.residual()

    def CoolProp_solve_outlet_fixed_P(self,mass_flow,p_in,T_in,p_out,iterate_on_enthalpy=False):
        """
        Solves for the outlet state given inlet conditions, outlet pressure,
        and mass flow rate using CoolProp for fluid properties.
        This method implements the numerical solution procedure described in
        Section 4 of Parisi et al.
        Args:
            mass_flow (float): Mass flow rate (kg/s).
            p_in (float): Inlet total pressure (Pa).
            T_in (float): Inlet total temperature (K).
            p_out (float): Outlet total pressure (Pa).
            iterate_on_enthalpy (bool): Use outlet enthalpy to compute state. If False, use temperature instead.
        """
        self._iterate_on_enthalpy=iterate_on_enthalpy
        self.mass_flow=mass_flow
        # Set inlet conditions
        self._fluid.update(cp.PT_INPUTS,p_in,T_in)
        self.inlet={"T":self._fluid.T(),
                    "p":self._fluid.p(),
                    "hmass"  :self._fluid.hmass(),
                    "smass"  :self._fluid.smass(),
                    "rhomass":self._fluid.rhomass(),
                    "viscosity" :self._fluid.viscosity(),
                    "speed_sound"  :self._fluid.speed_sound()}
        # Calculate isentropic outlet conditions
        self._fluid.update(cp.PSmass_INPUTS,p_out,self.inlet["smass"])
        self.outlet_isentropic={"T":self._fluid.T(),
                     "p":self._fluid.p(),
                     "hmass"  :self._fluid.hmass(),
                     "smass"  :self._fluid.smass(),
                     "rhomass"  :self._fluid.rhomass(),
                     "viscosity"  :self._fluid.viscosity(),
                     "speed_sound"  :self._fluid.speed_sound()}
        # Set bounds for Brent's method
        if iterate_on_enthalpy:
            x_low=self.outlet_isentropic["hmass"]
            x_in =self.inlet["hmass"]
        else:
            x_low=self.outlet_isentropic["T"]
            x_in =self.inlet["T"]
        # Determine  bounds for iteration based on machine type
        if p_out<p_in: #turbine -> real hmass or T is between inlet and isentropic
            x_high=x_in+(x_low-x_in)*self.reference.get("K_loweff",0.01)
        else: #compressor -> real hmass or T goes to inf as efficiency goes to 0
            x_high=x_in+(x_low-x_in)/self.reference["K_loweff"]
        
        x_out_solution,result=sp.brentq(self._CoolProp_iterate_outlet_fixed_P,x_low,x_high,full_output = True)
        residual=self._CoolProp_iterate_outlet_fixed_P(x_out_solution)
        if not result.converged:
            print("The root finding algorithm failed to find an outlet state that satisfies the efficiency correlation. Final resiudal: {residual:.2e}")
        return None

    def get_output_data(self):
        """Returns a copy of the calculated performance data."""
        if self._data is not None:
            return self._data.copy()
        return None

class CentrifugalCompressor(_Machine):
    """
    Non-dimensional model for centrifugal compressors.
    Based on Section 3.2 of Parisi et al.
    """       
    def __init__(self):
        self.data_in={"flow_coefficient":0.07,        # phi 
                      "backsweep":40.0,               # theta (degrees)
                      "is_shrouded":False,
                      "has_vaned":True,
                      "roughness":3.6e-6,             # R_a (m)
                      "clearance_height_ratio":0.02,  # (epsilon/b_2)
                      "metal_density":7800,           # rho_metal (kg/m^3)
                      "d_eta_other":0.0}
        self.reference={"diameter":0.4,               # D_ref (m)
                        "peripheral_speed":277,       # U_ref (m/s)
                        "roughness":3.6e-6,           # R_a_ref (m)
                        "clearance_height_ratio":0.02,# (epsilon/b_2)
                        "kinematic_viscosity":15.7e-6,# nu_ref (m^2/s)
                        "K_loweff":0.5,
                        "K_stress_unshrouded":0.41,
                        "K_stress_shrouded_1":0.67,
                        "K_stress_shrouded_2":0.75,
                        "phi_stress_shrouded_1":0.05,
                        "phi_stress_shrouded_2":0.12}
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
        dh =o["hmass"] - i["hmass"]
        dh_is = self.outlet_isentropic["hmass"] - i["hmass"]
        dh_p = dh - (o["smass"] - i["smass"]) * (o["T"] - i["T"]) / log(o["T"] / i["T"])
        eta_therm = dh_p / dh
        eta_is = dh_is / dh
        #import primary parameters
        phi = self.data_in["flow_coefficient"] #flow coefficient
        th = self.data_in["backsweep"]
        if phi<1e-4:
            raise ValueError("The selected value for the flow coefficient is <1e-4, this is outside the correlation range and will cause errors in the code")
        #baseline correlation
        if self.data_in["is_shrouded"]:
            k1 = +0.62
            k2 = -pi/4 * 0.4
            k3 = +pi/4 * 0.0014
            k4 = +0.51
            k5 = +4/pi * 1.0
            k6 = -(4/pi)**2 * 7.6
            k7 = -pi/4 * 0.00025
            phi_clipped = max(self.reference["K_stress_shrouded_phi1"], min(phi, self.reference["K_stress_shrouded_phi2"]))
            Kstress = self.reference["K_stress_shrouded_1"] + \
                    (self.reference["K_stress_shrouded_2"]-self.reference["K_stress_shrouded_1"]) * \
                    (phi_clipped-self.reference["phi_stress_shrouded_1"]) / \
                    (self.reference["phi_stress_shrouded_2"]-self.reference["phi_stress_shrouded_1"])
        else: #unshrouded
            k1 = +0.68
            k2 = -pi/4*0.37
            k3 = +pi/4*0.002
            k4 = +0.59
            k5 = +4/pi * 0.7
            k6 = -(4/pi)**2 * 7.5
            k7 = -pi/4 * 0.00025
            Kstress = self.reference["K_stress_unshrouded"]
        I_ref = k1 + (phi / k2)**3 + k3 / phi
        Ip_vaned = k4 + k5 * phi + k6 * phi**2 + k7 / phi
        eta_vaned=Ip_vaned / I_ref
        #vaneless correction
        if self.data_in["has_vaned"]:
            eta_base = eta_vaned
        else:
            eta_base = eta_vaned - 0.017 / (0.04 + (20/pi) * phi + eta_vaned**3)
        #work coefficient corrrection
        th_limited=min(th,40.0)
        d_eta_th = -0.0006 * (40.0 - th_limited)
        I = I_ref + (0.004 + 0.2 * phi**2) * (40.0 - th_limited)
        
        #Peripheral speed, Diameter, Mach
        U = sqrt(dh / I)
        D = sqrt(self.mass_flow / (i["rhomass"] * phi * U))
        Ma_u = U / i["speed_sound"]
        
        #Mach correction
        P_Ma = max(0, phi * (Ma_u - 0.8))
        d_eta_Ma = -(0.05 + 3 * P_Ma) * P_Ma
       
        # Reynolds number correction calculations
        # Relative velocity w/U
        w_over_U = 0.42 + 5 * phi - 14 * phi**2
        w = w_over_U * U
        # Diameter-to-chord ratio D/c
        Dc_ref = 1.1 + 22 * phi
        if th < 40.0:
            Dc = Dc_ref + 0.01 * (40.0 - th)
        else:
            Dc = Dc_ref
        c = D / Dc
        ksc=(1*self.data_in["roughness"])/c
        c_ref=c*(self.reference["diameter"]/D)
        # Relative roughness k_s/c
        ksc_ref = self.reference["roughness"] / c_ref
        
        # Reynolds number and friction factor
        kinematic_viscosity= i["viscosity"] / i["rhomass"]
        Re = w * c / kinematic_viscosity
        #The reference Reynolds number is indirectly a function of phi,
        #because it depends on U, therefore it is calculated by scaling 
        #with the length and peripeheral speed in the machine
        Re_ref = Re * (self.reference["diameter"]/D) * \
                      (self.reference["peripheral_speed"]/U) / \
                      (self.reference["kinematic_viscosity"]/kinematic_viscosity)
        
        f = friction_factor(Re, ksc)
        f_ref = friction_factor(Re_ref, ksc_ref)
        # Reynolds number correction
        d_eta_Re = -(0.05 + 0.002 / (phi + 0.0025)) * (f - f_ref) / f_ref
        
        # Clearance correction
        d_eta_cl = -0.5 * (self.data_in["clearance_height_ratio"] - self.reference["clearance_height_ratio"])

        eta_corr = eta_base + d_eta_th + d_eta_Ma + d_eta_Re + d_eta_cl + self.data_in["d_eta_other"]
        
        residual = eta_corr - eta_therm;
        self._data     ={"correlation_efficiency":eta_corr,
                         "polytropic_efficiency" :eta_therm,
                         "isentropic_efficiency" :eta_is,
                         "power"                 :dh*self.mass_flow,
                         "diameter"              :D,
                         "peripheral_speed"      :U,
                         "angular_speed"         :U/(D/2),
                         "mechanical_stress"     :Kstress*self.data_in["metal_density"]*U**2}
        
        return residual


class RadialTurbine(_Machine):
    """
    Non-dimensional model for radial inflow turbines.
    Based on Section 3.3 of Parisi et al.
    """
    def __init__(self):
        self.data_in={"specific_speed":0.6,            # oms
                      "is_scalloped":True,             #
                      "diffuser_loss_fraction":0.35,   #chi
                      "clearance_height_ratio":0.02,   #epsilon/b
                      "metal_density":7800,            # rho_metal (kg/m^3)
                      "d_eta_other":0.0}
        self.reference={"clearance_height_ratio":0.02, #epsilon/b
                        "Reynolds":352000.0,           #Re_ref
                        "K_loweff":0.1,
                        "K_stress_unshrouded":0.35,
                        "K_stress_scalloped":0.15}
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
        """Helper function to compute and store rotor outlet static conditions."""    
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
        dh_p=dh-(o["smass"]-i["smass"])*(o["T"]-i["T"])/log(o["T"]/i["T"]);
        eta_therm=dh/dh_is;
        eta_p=dh/dh_p;
        #import primary parameters
        oms=self.data_in["specific_speed"];
        #baseline correlation
        eta_base=0.93-0.3*(oms-0.6)**2-0.5*(oms-0.6)**3
        if oms<0.45:
            eta_base -= 4.2*(oms-0.45)**2
        kappa_rot=max(0.02+0.08*oms**2,0.2*oms**4)
        # Clearance correction 
        d_eta_cl = -1.15 * eta_base * (self.data_in["clearance_height_ratio"] - self.reference["clearance_height_ratio"])
        # Scallop correction 
        if self.data_in["is_scalloped"]:
            d_eta_scal=-0.03
            Kstress=self.reference['K_stress_scalloped']
        else:
            d_eta_scal=0.00
            Kstress=self.reference['K_stress_unshrouded']
        
        #overall efficiency
        eta_rot=eta_base+d_eta_cl+d_eta_scal+self.data_in["d_eta_other"]
        eta_overall=eta_rot/(1+self.data_in["diffuser_loss_fraction"]*kappa_rot)
        #Peripheral speed, Diameter, Mach
        nu=0.7
        dh_rot_is=dh_is/(1+self.data_in["diffuser_loss_fraction"]*kappa_rot)
        s_rot=i["smass"]+(o["smass"]-i["smass"])*(1/eta_rot-1)/(1/eta_overall-1)
        h_rot_static=o["hmass"]-kappa_rot*abs(dh_rot_is)
        s_rot_static=s_rot
        self.compute_rotor_outlet_static(h_rot_static,s_rot)
        U=nu*sqrt(2*(-dh_rot_is))
        om=oms*(-dh_rot_is)**0.75/(self.mass_flow/self.rotor_outlet_static["rhomass"])**0.5
        D=2*U/om
        Re=self.mass_flow/(i["viscosity"]*D/2);
        
        #Re correction
        eta_corr=1-(1-eta_overall)*(0.3+0.7*min(1,Re/self.reference["Reynolds"])**-0.2);

        
        residual=eta_corr-eta_therm;
        self._data     ={"correlation_efficiency":eta_corr,
                         "polytropic_efficiency" :eta_p,
                         "isentropic_efficiency" :eta_therm,
                         "power"                 :-dh*self.mass_flow,
                         "diameter"              :D,
                         "peripheral_speed"      :U,
                         "angular_speed"         :om,
                         "mechanical_stress"     :Kstress*self.data_in["metal_density"]*U**2}
        
        return residual



