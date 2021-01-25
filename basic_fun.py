import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

from shapely import geometry
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import MultiLineString
from shapely.ops import triangulate
from shapely.ops import cascaded_union

'''
parameters:
[[weight px py sa sb st]]
'''
'''
Update:
J(x+noise)
x = x - lr*J*noise   
'''

def init_fix_p(show=False):
  pts = np.array([[0,0],[0,2],[2.5,0]])
  f_pts = np.array([[2,2],[3,2]])
  f_mag = np.array([[0.5,1],[1,0.1]])
  all_pts = np.concatenate((pts,f_pts),axis=0)
  xmin,ymin = np.min(all_pts,axis=0)
  xmax,ymax = np.max(all_pts,axis=0)
  bounds = np.array([xmin,ymin,xmax,ymax])
  if show:
    fig = plt.figure(figsize=(6,6))
    plt.scatter(pts[:,0],pts[:,1],c='b')
    plt.scatter(f_pts[:,0],f_pts[:,1],c='r')
    for p,f in zip(f_pts,f_mag):
      plt.arrow(p[0],p[1],f[0],f[1])
  return pts,f_pts,f_mag,all_pts,bounds

def init_params(bounds,Nl=8):
  delta = bounds[2:]-bounds[:2]
  d_max = np.max(delta)/Nl
  x = np.arange(bounds[0],bounds[2]+d_max,d_max)
  y = np.arange(bounds[1],bounds[3]+d_max,d_max)
  params = np.ones((len(x)*len(y),6))

  idx = 0
  nx = len(x)
  tmp = np.ones_like(x)
  for j in y:
    params[idx:idx+nx,1]=x
    params[idx:idx+nx,2]=tmp*j
    idx+=nx
  
  params[:,-1]=np.zeros_like(params[:,-1])############################################
  p_bounds = np.ones((2,params.shape[0],params.shape[1]))*8
  p_bounds[0,:,:] *= -1 
  p_bounds[0,:,1:3] = params[:,1:3] - delta*0.1
  p_bounds[1,:,1:3] = params[:,1:3] + delta*0.1
  p_bounds[:,:,3:6] /=4
  p_bounds[:,:,-1]/=8/np.pi
  p_bounds[0,:,3:5]*=-1e-3
  
  return params,p_bounds

def gen_hull(params,bounds,Nl=16,show = False):
  delta = (bounds[2:]-bounds[:2])
  d_max = np.max(delta)/Nl
  x = np.arange(bounds[0]-delta[0]*0.25,bounds[2]+delta[0]*0.25+d_max,d_max)      #padding 0.2
  y = np.arange(bounds[1]-delta[1]*0.25,bounds[3]+delta[1]*0.25+d_max,d_max)      #padding 0.2

  x_1, y_1 = np.meshgrid(x, y) 
  xy = np.column_stack([x_1.flat, y_1.flat])
  
  tmp_z = np.zeros_like(x_1)

  for p in params: 
    mu = p[1:3]
    D = np.array([[p[3],0],[0,p[4]]]) #principle matrix
    R = np.array([[np.cos(p[5]),np.sin(p[5])],[-np.sin(p[5]),np.cos(p[5])]]) #rotation matrix
    cv = np.matmul(np.matmul(R.T,D),R)
    z = multivariate_normal.pdf(xy, mean=mu, cov=cv,allow_singular=True)
    z = z.reshape(x_1.shape)
    tmp_z+=p[0]*z

  tmp_max = np.max(tmp_z)
  fig, ax = plt.subplots()
  cs = ax.contourf(x_1, y_1, tmp_z,levels = [0,0.5,tmp_max+5])
  if show is False:
    plt.close() 

  if len(cs.collections[1].get_paths())>0:
    for ncp,cp in enumerate(cs.collections[1].get_paths()[-1].to_polygons()):
      lons = cp[:,0]
      lats = cp[:,1]
      new_shape = geometry.Polygon([(i[0], i[1]) for i in zip(lons,lats)])
      if ncp == 0:                
          poly = new_shape # first shape
      else:
          poly = poly.difference(new_shape) # Remove the holes
    return poly
  else:
    print('no contour is found for given params')
    return False

def pts_hull_check(hull,pts):
  '''
  weather points are inside hull check
  '''
  return hull.contains(MultiPoint(pts))

def performance(hull,sim_data,bounding_box,Y=0.5,C=2.5):
  box = bounding_box[2:]-bounding_box[:2]
  
  area = np.abs(hull.area/box[0]/box[1])   #ratio or percentage

  sigma = sim_data[2]
  sigma = np.concatenate((sigma[:,:2],np.zeros((len(sigma),1))),axis=1)
  tresca = np.max(sigma,axis=1)-np.min(sigma,axis=1)
  T = (tresca-Y)    
  T = np.sum(T*(T>0))/len(T)         #for normalization wrt nodes
  
  return C*T + area

def needs_refinement(vertices, area):
    max_area = 0.1
    return bool(area > max_area)

def triangulate_polygon(polygon,fix_pts, **kwargs):
    '''
    Given a shapely polygon, create a triangulation using meshpy.triangle
    
    Assume points are inside the polygon
 
    Parameters
    ---------
    polygon: Shapely.geometry.Polygon
    kwargs: passed directly to meshpy.triangle.build:
            triangle.build(mesh_info,
                           verbose=False,
                           refinement_func=None,
                           attributes=False,
                           volume_constraints=True,
                           max_volume=None,
                           allow_boundary_steiner=True,
                           allow_volume_steiner=True,
                           quality_meshing=True,
                           generate_edges=None,
                           generate_faces=False,
                           min_angle=None)
    Returns
    --------
    mesh_vertices: (n, 2) float array of 2D points
    mesh_faces:    (n, 3) int array of vertex indicies representing triangles
    '''
 
    if not polygon.is_valid:
        raise ValueError('invalid shapely polygon passed!')
 
 
    def round_trip(start, length):
        tiled = np.tile(np.arange(start, start + length).reshape((-1, 1)), 2)
        tiled = tiled.reshape(-1)[1:-1].reshape((-1, 2))
        tiled = np.vstack((tiled, [tiled[-1][-1], tiled[0][0]]))
        return tiled
 
    def add_boundary(boundary, start):

        coords = np.array(boundary.coords)
        unique = np.sort(trimesh.grouping.unique_rows(coords)[0])
        cleaned = coords[unique]
 
        vertices.append(cleaned)
        facets.append(round_trip(start, len(cleaned)))
        test = Polygon(cleaned)
        holes.append(np.array(test.representative_point().coords)[0])
 
        return len(cleaned)
 
    # sequence of (n,2) points in space
    vertices = deque()
    # sequence of (n,2) indices of vertices
    facets = deque()
    # list of (2) vertices in interior of hole regions
    holes = deque()
 
    start = add_boundary(polygon.exterior, 0)
    for interior in polygon.interiors:
        try:
            start += add_boundary(interior, start)
        except BaseException:
            log.warn('invalid interior, continuing')
            continue
    vertices = np.vstack(vertices)
    facets = np.vstack(facets).tolist() 
    f_flag = len(vertices)
    vertices = np.concatenate((vertices,fix_pts),axis=0)


    holes = np.array(holes)[1:]

    info = triangle.MeshInfo()
    info.set_points(vertices)
    info.set_facets(facets)
    info.set_holes(holes)
 
    # uses kwargs
    mesh = triangle.build(info, **kwargs)
 
    mesh_vertices = np.array(mesh.points)
    mesh_faces = np.array(mesh.elements)
 
    return mesh_vertices, mesh_faces,f_flag

def setup_files(all_pts,f_mag,mesh):
  nf = len(f_mag)
  nc = len(all_pts)-nf
  flag = mesh[2]

  mesh_points = mesh[0]
  mesh_tris = mesh[1]
  if len(mesh_points)<1 or nc<2 or nf<1:
    print('mesh is empty')
    return False
  else:
    nodal_int = np.arange(len(mesh_points)).reshape(-1,1)
    constrains = np.zeros(shape = mesh_points.shape)
    for c in range(nc):
      constrains[flag,:] = [-1,-1]
      flag+=1
    nodes = np.concatenate((nodal_int,mesh_points,constrains),axis=1) 
    np.savetxt('nodes.txt', nodes, fmt="%d  %f  %f  %d  %d") 

    ele_int = np.arange(len(mesh_tris)).reshape(-1,1)
    ele_type = np.ones(shape = (len(mesh_tris),2))*3
    ele_type[:,1] *= 0 
    eles = np.concatenate((ele_int,ele_type,mesh_tris),axis=1) 
    np.savetxt('eles.txt', eles, fmt="%d") 

    np.savetxt('mater.txt', [[10.0 ,0.3]], fmt="%f")

    loads = []
    for f in f_mag:
      loads.append([flag,f[0],f[1]])
      flag+=1
    np.savetxt('loads.txt', loads, fmt="%d %f %f") 
    return True