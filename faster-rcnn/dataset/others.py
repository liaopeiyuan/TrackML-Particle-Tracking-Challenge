from common import *
from sklearn import preprocessing


## processing df ###############################################

def get_particle_id(truth, volume_ids=None ,layer_ids=None, module_ids=None, is_remove_zero=True):

    t = truth
    if volume_ids is not None: t = t.loc[ t['volume_id'].isin(volume_ids) ]
    if layer_ids  is not None: t = t.loc[ t['layer_id' ].isin(layer_ids ) ]
    if module_ids is not None: t = t.loc[ t['module_id'].isin(module_ids) ]

    particle_ids = list(t['particle_id'].unique())
    if is_remove_zero:
        if 0 in particle_ids: particle_ids.remove(0)

    return particle_ids




## draw ###############################################

def draw_hit( ax, hits, marker='.', markersize=1, color=[0.75,0.75,0.75] ):
    x = hits.x.values
    y = hits.y.values
    z = hits.z.values
    ax.plot(x,y,z,marker,markersize=markersize,color=color)



def set_figure(ax, x_limit=[-1000,1000], y_limit=[-1000,1000], z_limit=[500,1000], title=''):

    ax.set_xlim(x_limit[0],x_limit[1]), ax.set_xlabel('x', fontsize=16)
    ax.set_ylim(y_limit[0],y_limit[1]), ax.set_ylabel('y', fontsize=16)
    ax.set_zlim(z_limit[0],z_limit[1]), ax.set_zlabel('z', fontsize=16)
    ax.set_title(title)
    #ax.axis('equal')
    #ax.set_aspect('equal')


## 3d gemoetry #########################################

def make_unit(d):
    d = d.copy()
    s = (d**2).sum(1)
    s = s**0.5
    d[:,0] /= s
    d[:,1] /= s
    d[:,2] /= s

    return d
