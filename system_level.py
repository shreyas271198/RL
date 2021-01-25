import numpy as np
import meshpy.triangle as triangle
import trimesh
from solidspy import solids_GUI

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
  j_params_list = zip(J_list, params_list)
  sorted_j_params_list = sorted(j_params_list, key = lambda x: x[0])  # take j value for sorting
  J_list, params_list = zip(*sorted_j_params_list)

  return params_list,J_list

def best_params(params_list,n_top=1):
  tmp_params = np.stack(params_list[:n_top],axis=0).mean(axis=0)
  tmp_params = params_check(tmp_params)
  J = system(tmp_params)
  if J is not False:
    print('J value is : ', J)
    return tmp_params
  else:
    print('not updated')
    return np.copy(params_list[0])

def sigma_update(sigma,decay_rate=0.98):
  return sigma*decay_rate
