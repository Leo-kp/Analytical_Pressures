from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy import special as sc

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



def pwd_vf_ir(
    table_p,
    table_d,
    units:str="US"    
):
   #based upon US, it converst to SI internally k and xf
   k=np.linspace(0.1,100,100) #shape(M,) --> when broadcasted (1,M)
   xf=np.linspace(100,10000)
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