from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy import special as sc 
import scipy.optimize as opt


def conv_eng_si(value,
               unit:str
):
  #si/eng
  factor={'psi': 6894.76,
          'psi-1': 1/6894.76,
          'acres': 4046.86,
          'ft': 0.3048,
          'h': 3600,
          'cP': 0.001,
          'RB/STB':1,
          'STB/D': 0.0000018401,
          'vol_fraction':1,
          'md':9.8692326671601E-16
  }

  unit_c={'psi': 'Pa',
          'psi-1': 'Pa-1',
          'acres': 'm**2',
          'ft': 'm',
          'h': 's',
          'cP': 'Pa*s',
          'RB/STB': ' ',
          'STB/D': 'm**3/s',
          'vol_fraction':'vol_fraction',
          'md':'m**2'
  }

  try:
    conversion_factor = factor[unit]
    converted_value = value * conversion_factor
        
    return converted_value,unit_c[unit]
        
  except KeyError:
    print(f"Error: Unit '{unit}' not found in conversion dictionary.")


#-----------------------------------------------------------------------------
def pws_vw_ir(
    table_p,
    table_d,
    k
):
  
   h=table_p.loc[table_p['Variable'] == 'h', 'Value'].iloc[0]
   pi=table_p.loc[table_p['Variable'] == 'p_i', 'Value'].iloc[0]
   q=table_p.loc[table_p['Variable'] == 'q', 'Value'].iloc[0]
   B=table_p.loc[table_p['Variable'] == 'B', 'Value'].iloc[0]
   mu=table_p.loc[table_p['Variable'] == 'mu', 'Value'].iloc[0]
   t=table_p.loc[table_p['Variable'] == 't', 'Value'].iloc[0]
   por=table_p.loc[table_p['Variable'] == 'por', 'Value'].iloc[0]
   c_t=table_p.loc[table_p['Variable'] == 'c_t', 'Value'].iloc[0] 
   r=table_p.loc[table_p['Variable'] == 'r_w', 'Value'].iloc[0] 
  
   dt=table_d["delta_t"]
   t_dt=table_d["t_dt"]

   pws=pi+( (q*mu*B/(4*np.pi*h*k))*(sc.expi(-por*mu*c_t*(r**2)/(4*k*(t+dt))) - sc.expi(-por*mu*c_t*(r**2)/(4*k*dt))))

   return pws

def pws_vw_ira(
    table_p,
    table_d,
    k
):
  
   h=table_p.loc[table_p['Variable'] == 'h', 'Value'].iloc[0]
   pi=table_p.loc[table_p['Variable'] == 'p_i', 'Value'].iloc[0]
   q=table_p.loc[table_p['Variable'] == 'q', 'Value'].iloc[0]
   B=table_p.loc[table_p['Variable'] == 'B', 'Value'].iloc[0]
   mu=table_p.loc[table_p['Variable'] == 'mu', 'Value'].iloc[0]
   t=table_p.loc[table_p['Variable'] == 't', 'Value'].iloc[0]
  
   dt=table_d["delta_t"]
   t_dt=table_d["t_dt"]

   pws=pi- (q*mu*B/(4*np.pi*h*k))*np.log(t_dt)

   return pws


#----------------------------------------------------------------------------

def pws_vw_bcr(
    table_p,
    table_d,
    k:float,
    N_terms:int=50

):
  
   h=table_p.loc[table_p['Variable'] == 'h', 'Value'].iloc[0]
   pi=table_p.loc[table_p['Variable'] == 'p_i', 'Value'].iloc[0]
   q=table_p.loc[table_p['Variable'] == 'q', 'Value'].iloc[0]
   B=table_p.loc[table_p['Variable'] == 'B', 'Value'].iloc[0]
   mu=table_p.loc[table_p['Variable'] == 'mu', 'Value'].iloc[0]
   t=table_p.loc[table_p['Variable'] == 't', 'Value'].iloc[0]
   por=table_p.loc[table_p['Variable'] == 'por', 'Value'].iloc[0]
   c_t=table_p.loc[table_p['Variable'] == 'c_t', 'Value'].iloc[0] 
   r=table_p.loc[table_p['Variable'] == 'r_w', 'Value'].iloc[0] 
   R_b=table_p.loc[table_p['Variable'] == 'r_e', 'Value'].iloc[0] 
  
   dt=table_d["delta_t"]
   tdt=table_d["tdt"]

   X_n = sc.jn_zeros(1, N_terms)
   X_n_col = X_n[:, np.newaxis]
      
   alpha= k / (por * mu * c_t * (R_b**2))
   alpha_n_col = alpha * (X_n_col**2)
    
   J0_rw = sc.jn(0, X_n * r / R_b)
   J0_X = sc.jn(0, X_n)
   Denominator = (X_n**2) * (J0_X**2)
   C_n_col = (J0_rw / Denominator)[:, np.newaxis]

   def p_Yt(tf):

        exp_argument_matrix = -1 * alpha_n_col * tf
        Exp_matrix = np.exp(exp_argument_matrix)

        S_t_sum_array = np.sum(C_n_col * Exp_matrix, axis=0)

        Yt= 4*k*tf/(por*mu*c_t*R_b**2) + r**2/R_b**2 -2*np.log(r/R_b) -3/2 -4*S_t_sum_array
        return Yt
   
   tdt_np = tdt.values # or tdt.to_numpy()
   dt_np = dt.values # or dt.to_numpy()
   pws= pi - q*mu*B/(4*np.pi*k*h)*(p_Yt(tdt_np) - p_Yt(dt_np))
   return pws

#--------------------------------------------------------------------------------------------------

def pws_vw_bfcr(
    table_p,
    table_d,
    k:float,
    N_terms:int=50

):
  
   h=table_p.loc[table_p['Variable'] == 'h', 'Value'].iloc[0]
   pi=table_p.loc[table_p['Variable'] == 'p_i', 'Value'].iloc[0]
   q=table_p.loc[table_p['Variable'] == 'q', 'Value'].iloc[0]
   B=table_p.loc[table_p['Variable'] == 'B', 'Value'].iloc[0]
   mu=table_p.loc[table_p['Variable'] == 'mu', 'Value'].iloc[0]
   t=table_p.loc[table_p['Variable'] == 't', 'Value'].iloc[0]
   por=table_p.loc[table_p['Variable'] == 'por', 'Value'].iloc[0]
   c_t=table_p.loc[table_p['Variable'] == 'c_t', 'Value'].iloc[0] 
   r=table_p.loc[table_p['Variable'] == 'r_w', 'Value'].iloc[0] 
   R_b=table_p.loc[table_p['Variable'] == 'r_e', 'Value'].iloc[0] 
  
   dt=table_d["delta_t"]
   tdt=table_d["tdt"]

   def characteristic_eq(Xn, rde):
        J1_Xn = sc.jn(1, Xn)
        Y1_Xn_rde = sc.yn(1, Xn * rde)
        Y1_Xn = sc.yn(1, Xn)
        J1_Xn_rde = sc.jn(1, Xn * rde)
        
        return Y1_Xn * J1_Xn_rde -J1_Xn * Y1_Xn_rde
  
   rde = R_b / r 
   X_n = np.zeros(N_terms)
   root_counter = 0
   
   step_est = np.pi / (rde - 1)
   search_start = step_est * 0.1 
   search_step = step_est * 0.8 

   while root_counter < N_terms:
    a = search_start
    b = search_start + search_step

    if characteristic_eq(a, rde) * characteristic_eq(b, rde) < 0:
        try:
            result = opt.root_scalar(characteristic_eq, args=(rde,), bracket=[a, b], method='brentq')
            
            if result.converged:
                X_n[root_counter] = result.root
                root_counter += 1
                search_start = result.root + step_est * 0.1 
            else:
                search_start = b
                
        except ValueError:
            search_start = b
            
    else:
        search_start = b

    if search_start > N_terms * np.pi: # Heuristic limit
        print(f"Warning: Could not find all {N_terms} roots. Found {root_counter}.")
        X_n = X_n[0:root_counter] 
        break

    if search_start > 1000: break

   X_n_col = X_n[:, np.newaxis]
      
   alpha= k / (por * mu * c_t * (r**2))
   alpha_m_col = alpha * (X_n_col**2)
    
   J1_rb = sc.jn(1, X_n * R_b/r)
   J1_X = sc.jn(1, X_n)

   Denominator = (X_n**2) * ((J1_rb**2)-(J1_X**2))
   C_n_col = (J1_rb**2 / Denominator)[:, np.newaxis]

   def p_Yt(tf):

        exp_argument_matrix = -1 * alpha_m_col * tf
        Exp_matrix = np.exp(exp_argument_matrix)

        term_1= 4*( alpha* tf + 1/4)/(rde**2 -1)
        term_2= (3*rde**4-4*rde**4*np.log(rde)-2*rde**2-1) /(2*(rde**2-1)**2) 
        S_t_sum_array = np.sum(C_n_col * Exp_matrix, axis=0)

        Yt= term_1 - term_2 + 4*S_t_sum_array

        return Yt
   
   tdt_np = tdt.values # or tdt.to_numpy()
   dt_np = dt.values # or dt.to_numpy()
   pws= pi - q*mu*B/(4*np.pi*k*h)*(p_Yt(tdt_np) - p_Yt(dt_np))
   return pws


#----------------------------------------------------------------------------------------------------

def pws_vw_bpcr(
    table_p,
    table_d,
    k:float,
    N_terms:int=50

):
  
   h=table_p.loc[table_p['Variable'] == 'h', 'Value'].iloc[0]
   pi=table_p.loc[table_p['Variable'] == 'p_i', 'Value'].iloc[0]
   q=table_p.loc[table_p['Variable'] == 'q', 'Value'].iloc[0]
   B=table_p.loc[table_p['Variable'] == 'B', 'Value'].iloc[0]
   mu=table_p.loc[table_p['Variable'] == 'mu', 'Value'].iloc[0]
   t=table_p.loc[table_p['Variable'] == 't', 'Value'].iloc[0]
   por=table_p.loc[table_p['Variable'] == 'por', 'Value'].iloc[0]
   c_t=table_p.loc[table_p['Variable'] == 'c_t', 'Value'].iloc[0] 
   r=table_p.loc[table_p['Variable'] == 'r_w', 'Value'].iloc[0] 
   R_b=table_p.loc[table_p['Variable'] == 'r_e', 'Value'].iloc[0] 
  
   dt=table_d["delta_t"]
   #tdt=table_d["tdt"]
   tdt= dt+t

   def characteristic_eq(Xm, rde):
        J1_Xm = sc.jn(1, Xm)
        Y0_Xm_rde = sc.yn(0, Xm * rde)
        Y1_Xm = sc.yn(1, Xm)
        J0_Xm_rde = sc.jn(0, Xm * rde)
        
        return J1_Xm * Y0_Xm_rde - Y1_Xm * J0_Xm_rde

   rde = R_b / r 
   X_m = np.zeros(N_terms)

   root_counter = 0
   
   step_est = np.pi / (rde - 1)
   search_start = step_est * 0.1 
   search_step = step_est * 0.8 

   while root_counter < N_terms:
    a = search_start
    b = search_start + search_step

    if characteristic_eq(a, rde) * characteristic_eq(b, rde) < 0:
        try:
            result = opt.root_scalar(characteristic_eq, args=(rde,), bracket=[a, b], method='brentq')
            
            if result.converged:
                X_m[root_counter] = result.root
                root_counter += 1
                search_start = result.root + step_est * 0.1  
            else:
                search_start = b
                
        except ValueError:
            search_start = b
            
    else:
        search_start = b

    if search_start > N_terms * np.pi: # Heuristic limit
        print(f"Warning: Could not find all {N_terms} roots. Found {root_counter}.")
        X_m = X_m[0:root_counter] 
        break
    if search_start > 1000: break

   X_m_col = X_m[:, np.newaxis]
      
   alpha= k / (por * mu * c_t * (r**2))
   alpha_m_col = alpha * (X_m_col**2)
    
   J0_rb = sc.jn(0, X_m * R_b/r)
   J1_X = sc.jn(1, X_m)

   Denominator = (X_m**2) * ((J1_X**2)-(J0_rb**2))
   C_n_col = (J0_rb**2 / Denominator)[:, np.newaxis]

   def p_Yt(tf):

        exp_argument_matrix = -1 * alpha_m_col * tf
        Exp_matrix = np.exp(exp_argument_matrix)

        S_t_sum_array = np.sum(C_n_col * Exp_matrix, axis=0)

        Yt= 2*np.log(R_b/r) -4*S_t_sum_array

        return Yt
   
   tdt_np = tdt.values # or tdt.to_numpy()
   dt_np = dt.values # or dt.to_numpy()
   pws= pi - q*mu*B/(4*np.pi*k*h)*(p_Yt(tdt_np) - p_Yt(dt_np))
   return pws

#------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------

def pwd_vf_ir(
    table_p,
    table_d,
    units:str="US"    
):
   #based upon US, it converst to SI internally k and xf
  #  k=np.linspace(0.1,100,100) #shape(M,) --> when broadcasted (1,M)
  #  xf=np.linspace(100,10000)
   k=np.logspace(-2,2) #shape(M,) --> when broadcasted (1,M)
   start=np.log10(2)
   end=np.log10(2000)
   xf=np.linspace(start,end,num=10)
   xf_c = xf[:, np.newaxis] #reshaping for vector application (N,1)

   h=table_p.loc[table_p['Variable'] == 'h', 'Value'].iloc[0]
   pi=table_p.loc[table_p['Variable'] == 'p_i', 'Value'].iloc[0]
   pwf=table_d.loc[0,'pws']
   q=table_p.loc[table_p['Variable'] == 'q', 'Value'].iloc[0]
   B=table_p.loc[table_p['Variable'] == 'B', 'Value'].iloc[0]
   mu=table_p.loc[table_p['Variable'] == 'mu', 'Value'].iloc[0]
   t=table_p.loc[table_p['Variable'] == 't', 'Value'].iloc[0]
   por=table_p.loc[table_p['Variable'] == 'por', 'Value'].iloc[0]
   c_t=table_p.loc[table_p['Variable'] == 'c_t', 'Value'].iloc[0] 
   
   if units=="US":
    pwd=k*h*(pi-pwf)/(141.2*q*B*mu)
    td=(0.000264)*k*t/(por*mu*c_t*(xf_c**2)) #final shape (N,M) 
   elif units=="SI":  #consider rid-off internal conversion when needed
    k=k*9.8692326671601E-16
    xf_c=xf_c*0.3048
    pwd=2*np.pi*k*h*(pi-pwf)/(q*B*mu)
    td=k*t/(por*mu*c_t*(xf_c**2)) #final shape (N,M) 

   return pwd,td
   
#--------------------------------------------------------------------


def pwd_vf_uf(
    table_p,
    table_d,
    units:str="US"    
):
  
   #based upon US, it converst to SI internally k and xf
   #k=np.linspace(0.1,100,10) #shape(M,) --> when broadcasted (1,M)
   k=np.logspace(start=-1, stop=3, num=10)
   A=table_p.loc[table_p['Variable'] == 'A', 'Value'].iloc[0]
   #xe=math.sqrt(A)/2
   #xf=np.linspace(2,xe,7)
   start_exp = np.log10(2)
   stop_exp = np.log10(2000)
   xf=np.logspace(start=start_exp, stop=stop_exp, num=10)
   xf_c = xf[:, np.newaxis] #reshaping for vector application (N,1)


   h=table_p.loc[table_p['Variable'] == 'h', 'Value'].iloc[0]
   pi=table_p.loc[table_p['Variable'] == 'p_i', 'Value'].iloc[0]
   pwf=table_d.loc[0,'pws']
   q=table_p.loc[table_p['Variable'] == 'q', 'Value'].iloc[0]
   B=table_p.loc[table_p['Variable'] == 'B', 'Value'].iloc[0]
   mu=table_p.loc[table_p['Variable'] == 'mu', 'Value'].iloc[0]
   t=table_p.loc[table_p['Variable'] == 't', 'Value'].iloc[0]
   por=table_p.loc[table_p['Variable'] == 'por', 'Value'].iloc[0]
   c_t=table_p.loc[table_p['Variable'] == 'c_t', 'Value'].iloc[0] 
   
   if units=="US":
    td=(0.000264)*k*t/(por*mu*c_t*(xf_c**2)) #final shape (N,M) 
    pwd=np.sqrt(np.pi*td)*sc.erf(1/(2*np.sqrt(td))) - 0.5*sc.expi(-1/(4*td))
   elif units=="SI":  #consider rid-off internal conversion when needed
    k=k*9.8692326671601E-16
    # xf_c=xf_c*0.3048
    td=k*t/(por*mu*c_t*(xf_c**2))
    pwd=np.sqrt(np.pi*td)*sc.erf(1/(2*np.sqrt(td))) - 0.5*sc.expi(-1/(4*td))
 
   return pwd,td


def pwd_vf_ic(
    table_p,
    table_d,
    units:str="US"    
):
  
   #based upon US, it converst to SI internally k and xf
   #k=np.linspace(0.1,100,10) #shape(M,) --> when broadcasted (1,M)
   k=np.logspace(start=-1, stop=3, num=10)
   A=table_p.loc[table_p['Variable'] == 'A', 'Value'].iloc[0]
   #xe=math.sqrt(A)/2
   #xf=np.linspace(2,xe,7)
   start_exp = np.log10(2)
   stop_exp = np.log10(2000)
   xf=np.logspace(start=start_exp, stop=stop_exp, num=10)
   xf_c = xf[:, np.newaxis] #reshaping for vector application (N,1)


   h=table_p.loc[table_p['Variable'] == 'h', 'Value'].iloc[0]
   pi=table_p.loc[table_p['Variable'] == 'p_i', 'Value'].iloc[0]
   pwf=table_d.loc[0,'pws']
   q=table_p.loc[table_p['Variable'] == 'q', 'Value'].iloc[0]
   B=table_p.loc[table_p['Variable'] == 'B', 'Value'].iloc[0]
   mu=table_p.loc[table_p['Variable'] == 'mu', 'Value'].iloc[0]
   t=table_p.loc[table_p['Variable'] == 't', 'Value'].iloc[0]
   por=table_p.loc[table_p['Variable'] == 'por', 'Value'].iloc[0]
   c_t=table_p.loc[table_p['Variable'] == 'c_t', 'Value'].iloc[0] 
   
   if units=="US":
    td=(0.000264)*k*t/(por*mu*c_t*(xf_c**2)) #final shape (N,M) 
    pwd=np.sqrt(np.pi*td)*sc.erf(1/(2*np.sqrt(td))) - 0.5*sc.expi(-1/(4*td))
   elif units=="SI":  #consider rid-off internal conversion when needed
    k=k*9.8692326671601E-16
    # xf_c=xf_c*0.3048
    td=k*t/(por*mu*c_t*(xf_c**2))
    pwd=np.sqrt(np.pi*td)*sc.erf(1/(2*np.sqrt(td))) - 0.5*sc.expi(-1/(4*td))
 
   return pwd,td