import numpy as np
import meshpy.triangle as triangle
import trimesh
from solidspy import solids_GUI
from .basic_fun import *

def globle_vars():
  pts,f_pts,f_mag,all_pts,bounds = init_fix_p(show=False) 
  params,p_bounds = init_params(bounds,Nl=8)
  return pts,f_pts,f_mag,all_pts,bounds,params,p_bounds

def system(params):
  params_tmp = np.copy(params)
  poly = gen_hull(params_tmp,BOUNDS,Nl=16)  
  if poly is not False:
    if pts_hull_check(poly,ALL_PTS):
      mesh = triangulate_polygon(poly,ALL_PTS,refinement_func=needs_refinement)
      if setup_files(ALL_PTS,F_MAG,mesh):
        try:
          sim_data = solids_GUI(plot_contours=False,compute_strains=True, folder='') 
          J = performance(poly,sim_data,BOUNDS,Y=0.5,C=2.5)
          return J
        except:
          print('simulation error occured : skipping this simulation step')
          return False
      else:
        print('setup files failed')
        return False
    else:
      print('pts not in hull')
      return False
  else:
    print('failed to generate hull')
    return False

def params_check(params):
  params = np.clip(params,P_BOUNDS[0],P_BOUNDS[1])
  return params

def params_noise(params,sigma=1,pop_size=10):
  params_list = []
  J_list = []
  #generating noise in params
  while len(params_list)<pop_size:
    params_tmp = params + np.random.randn(*params.shape)*sigma 
    params_tmp = params_check(params_tmp)
    J = system(params_tmp)
    #output.clear()
    if J is not False:
      params_list.append(params_tmp)
      J_list.append(J)

  #sorting according to J 
  j_params_
