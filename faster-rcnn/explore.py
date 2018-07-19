import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2' #'3,2,1,0'

## my library ---
from common import *
from draw import *
from utility.file import *
from utility.draw import *
from net.rate   import *
from net.metric import *

INF=3000
## other library ---
from dataset.trackml.score  import score_event
from dataset.others import *
from sklearn.cluster.dbscan_ import dbscan
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

REMAP_LAYER_ID ={
    (12,  2): 0,   (16,  2): 0,    (14,  2): 0,   (18,  2): 0,
    (12,  4): 1,   (16,  4): 1,    (14,  4): 1,   (18,  4): 1,
    (12,  6): 2,   (16,  6): 2,    (14,  6): 2,   (18,  6): 2,
    (12,  8): 3,   (16,  8): 3,    (14,  8): 3,   (18,  8): 3,
    (12, 10): 4,   (16, 10): 4,    (14, 10): 4,   (18, 10): 4,
    (12, 12): 5,   (16, 12): 5,    (14, 12): 5,   (18, 12): 5,
    (12, 14): 6,   (16, 14): 6,    (14, 14): 6,   (18, 14): 6,

    ( 7,  2): 0,   ( 9,  2): 0,
    ( 7,  4): 1,   ( 9,  4): 1,
    ( 7,  6): 2,   ( 9,  6): 2,
    ( 7,  8): 3,   ( 9,  8): 3,
    ( 7, 10): 4,   ( 9, 10): 4,
    ( 7, 12): 5,   ( 9, 12): 5,
    ( 7, 14): 6,   ( 9, 14): 6,
    ( 7, 16): 7,   ( 9, 16): 7,

    (13,  2): 0,
    (13,  4): 1,
    (13,  6): 2,
    (13,  8): 3,
    (17,  2): 4,
    (17,  4): 5,

    ( 8,  2): 0,
    ( 8,  4): 1,
    ( 8,  6): 2,
    ( 8,  8): 3
}
LAYER_NUM = 7

# http://code.activestate.com/recipes/578112-bresenhams-line-algorithm-in-n-dimensions/
# http://members.chello.at/easyfilter/bresenham.html
# from utility.bresenhams import bresenhamline



def point_to_data(
    a, zr, z, layer_id, p,
    a_limit, zr_limit,
    D,H,W
):
    a0, a1  = a_limit
    zr0,zr1 = zr_limit

    idx = np.where((a>a0) & (a<a1) & (zr>zr0) & (zr<zr1))[0]
    a  = a [idx]
    zr = zr[idx]
    p  = p [idx]
    layer_id = layer_id[idx]

    # digitalize into voxels ...
    ii = np.floor(( a- a0)/( a1- a0)*W).astype(np.int32)
    ij = np.floor((zr-zr0)/(zr1-zr0)*H).astype(np.int32)
    ik = layer_id

    image = np.zeros((D,H,W), np.uint8)
    image[ik,ij,ii] = 1


    # build neighbourhood graph ...
    pair = []
    link = []
    for d in range(D-1):
        l0 = np.where(layer_id==d  )[0]
        l1 = np.where(layer_id==d+1)[0]
        if (len(l0)==0) | (len(l1)==0) : continue


        q1  = np.column_stack((a[l1]*5,zr[l1]))
        q0  = np.column_stack((a[l0]*5,zr[l0]))
        kdt = KDTree(q1)  #<todo> fix angular discontinuinty
        n   = kdt.query_radius(q0, 0.25) #2

        L0 = len(q0)
        for j in range(L0):
            i0 = l0[j]
            u0 = p[i0]
            #aa0, rr0, zzr0 = q0[i0]

            L1 = len(n[j])
            for i in range(L1):
                i1 = l1[n[j][i]]
                u1 = p[i1]
                #aa1, rr1, zzr1 = q1[i1]

                #print(u0,u1)
                l = (u1!=0) & (u1==u0) #k==0 #

                pair.append((i0,i1))
                link.append(l)

    link = np.array(link)
    pair = np.array(pair)

    if (len(pair)==0):
        print(a_limit,zr_limit)

    #----
    p0 = pair[:,0]
    p1 = pair[:,1]
    i0 = np.floor(( a[p0]- a0)/( a1- a0)*W).astype(np.int32)
    j0 = np.floor((zr[p0]-zr0)/(zr1-zr0)*H).astype(np.int32)
    k0 = layer_id[[p0]]
    i1 = np.floor(( a[p1]- a0)/( a1- a0)*W).astype(np.int32)
    j1 = np.floor((zr[p1]-zr0)/(zr1-zr0)*H).astype(np.int32)
    k1 = layer_id[[p1]]

    location = ( k0,j0,i0,  k1,j1,i1 )
    location = np.column_stack(location)

    pair = idx[pair]

    return image, location, pair, link









##########################################################33


def run_study_cnn ():

    # event_ids = ['000001030', '000001029', '000001028', '000001027', '000001026', '000001025'] #
    # event_id = '000001001'

    event_ids = ['%09d'%i for i in range(1000,1050)]


    for event_id in event_ids:

        save_dir  = '/home/alexanderliao/data/Kaggle/competitions/trackml-particle-identification/results'
        data_dir  = '/home/alexanderliao/data/Kaggle/competitions/trackml-particle-identification/train_triplet'
        particles = pd.read_csv(data_dir + '/event%s-particles.csv'%event_id)
        hits      = pd.read_csv(data_dir + '/event%s-hits.csv' %event_id)
        truth     = pd.read_csv(data_dir + '/event%s-truth.csv'%event_id)

        truth = truth.merge(hits,       on=['hit_id'],      how='left')
        truth = truth.merge(particles,  on=['particle_id'], how='left')


        #----------------------------
        df = truth  #.copy()
        #df = df.assign(my_layer_id  = df.layer_id.replace(REMAP_LAYER_ID, inplace=False))

        df = df.assign(r   = np.sqrt( df.x**2 + df.y**2))
        df = df.assign(d   = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
        df = df.assign(a   = np.arctan2(df.y, df.x))
        df = df.assign(cosa= np.cos(df.a))
        df = df.assign(sina= np.sin(df.a))
        df = df.assign(zr  = df.z/df.r)
        df = df.assign(phi = np.arctan2(df.z, df.r))
        df = df.assign(momentum = np.sqrt( df.px**2 + df.py**2 + df.pz**2 ))
        df = df.assign(vel      = np.sqrt( df.vx**2 + df.vy**2 ))
        df.loc[df.particle_id==0,'momentum']=0

        #top volume_id = (14,18,);  layer_id = (2, 4, 6, 8, 10, 12,)
        my_layers = [0,1,2,3,4,5,]
        df = df.loc[(df.z>1100)& (df.z<INF)]
        df = df.loc[(df.r> 200)& (df.r<INF)]

        #-------------------------------------------------------

        #### 1. select sub volume ##############################
        if 0:
            a0,  a1  = -1.2, -0.2
            zr0, zr1 =  4.5,  6.5
            z0,  z1  = 1200, 3200
            D,H,W    = 8,512,512

        if 0:
            a0,  a1  = -1.2, -0.8
            zr0, zr1 =  4.5,  6.5
            z0,  z1  = 1200, 3200
            D,H,W    = 8,512,512

        if 0:
            a0,  a1  = -3.2,  3.2
            zr0, zr1 =    0, 12.8
            z0,  z1  = 1200, 3200
            D,H,W    = 8,6400,3200

        if 1:
            a0,  a1  =  -3.2,  3.2
            zr0, zr1 =     0, 12.8
            z0,  z1  =  1200, 3200
            D,H,W    = 8,3200,1600



        df = df.loc[(df.a >a0 ) & (df.a <a1 )]
        df = df.loc[(df.z >z0 ) & (df.z <z1 )]
        df = df.loc[(df.zr>zr0) & (df.zr<zr1)]
        N  = len(df)
        df = df.assign(my_layer_id  = df[['volume_id','layer_id']].apply(lambda x: REMAP_LAYER_ID[tuple(x)], axis=1))
        my_layer_id = df['my_layer_id'].values.astype(np.int32)
        momentum    = df['momentum'].values.astype(np.float32)
        velocity    = df['vel'].values.astype(np.float32)
        hit_id      = df['hit_id'].values.astype(np.int64)
        p           = df['particle_id'].values.astype(np.int64)
        x,y,z,r,a,cosa,sina,phi,zr = df[['x', 'y', 'z', 'r', 'a', 'cosa', 'sina', 'phi', 'zr']].values.astype(np.float32).T

        particle_ids = np.unique(p)
        particle_ids = particle_ids[particle_ids!=0]
        num_particle_ids = len(particle_ids)

        #colors = plt.cm.gnuplot( my_layer_id/7 )  #hsv   #gnuplot    #jet
        colors = plt.cm.gnuplot(np.arange(D)/D),



        if 0:  ### just show ##############################################################

            AX3d1.clear()
            plot3d_particles(AX3d1, particle_ids,p, a,zr,z, z, subsample=1)
            AX3d1.scatter(a, zr,  z, c=plt.cm.gnuplot( my_layer_id/D ), s=16, edgecolors='none')

            AX1.clear()
            plot_particles(AX1, particle_ids,p, a, zr, z, subsample=1)
            AX1.scatter(a, zr, c=plt.cm.gnuplot( my_layer_id/D ), s=16, edgecolors='none')

            plt.show()



        #### show results ####
        if 0:
            #image, location, pair, link = point_to_data( a, zr, z, my_layer_id, p, (a0,a1), (zr0,zr1), D,H,W )

            #---
            AX3d2.clear()
            AX3d2.set_aspect('equal')
            draw_voxel(AX3d2,image,colors=plt.cm.gnuplot( np.arange(D)/D ),markersize=10)
            #set_axes_equal(AX3d2)


            L = len(location)
            for i in range(0,L,10):
                l = link[i]
                k0, j0, i0, k1, j1, i1 = location[i]

                if l==1:
                    color=[0,0,0]  #np.random.uniform(1,0,3)  #[0,0,0]
                    AX3d2.plot([i0,i1],[j0,j1],[k0,k1],'-', color=color,markersize=1)

            plt.show()
            exit(0)


        #plt.show()
        save_pickle_file(save_dir+'/data_1600.%s.3dcnn.npy'%event_id,(a, zr, z, my_layer_id, p))
        print('saved %s'%event_id)



    # plt.show()



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_study_cnn()


#  ffmpeg -f image2  -pattern_type glob -r 33 -i "iterations/*.png" -c:v libx264  iterations.mp4
#  convert *.png animated.gif
#
