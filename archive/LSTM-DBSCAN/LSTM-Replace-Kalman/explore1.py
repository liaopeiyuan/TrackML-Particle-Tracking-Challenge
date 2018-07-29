from common import *
from utility.file import *


from dataset.trackml.dataset import load_event, load_dataset
from dataset.trackml.randomize import shuffle_hits
from dataset.trackml.score import score_event
from dataset.others import *


#------------------------------------------------------




def run_explore():

    data_dir  = '/mydisk/TrackML-Data'
    detectors = pd.read_csv(data_dir + '/detectors.csv')

    events = [
        '000001025','000001026','000001027','000001028','000001029',#
        '000001030','00001031','000001032','000001033','000001034',
    ]
    events = glob.glob(data_dir+'/train_2/event*-truth.csv')
    sorted(events)
    events = [e.split('/')[-1].replace('event','').replace('-truth.csv','') for e in events]
    #events = ['000001093']

    #https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(111, projection='3d')

    samples=[]
    for event in events:
        print(event)
        ax.clear()

        particles = pd.read_csv(data_dir + '/train_2/event%s-particles.csv'%event)
        hits   = pd.read_csv(data_dir + '/train_2/event%s-hits.csv'%event)
        truth  = pd.read_csv(data_dir + '/train_2/event%s-truth.csv'%event)
        truth  = truth.merge(hits, on=['hit_id'], how='left')

        # ----------------

        h = truth.loc[
              (truth.volume_id.isin([9,])) & (truth.layer_id.isin([2,4,6,]))  & (truth.module_id.isin([1,]))
        ]
        num_hits =len(h)
        hit_id = h.hit_id.values
        particle_id = h.particle_id.values
        x = h.x.values
        y = h.y.values
        z = h.z.values
        layer_id = h.layer_id.values

        invalid_ids =[]
        for p in np.unique(particle_id):
            if p==0: continue
            L = (particle_id==p).sum()
            #print(L)

            if L!=3:
                invalid_ids.append(p)
                continue

            # select only straight tracks for initial experiments ------------------
            idx = np.where(particle_id==p)[0]
            xx = x[idx]
            yy = y[idx]
            zz = z[idx]

            dis = (xx**2 + yy**2 + zz**2)
            idx = dis.argsort()
            xx = xx[idx]
            yy = yy[idx]
            zz = zz[idx]

            grad = np.vstack([
                np.array([xx[0],yy[0],zz[0]]),
                np.array([
                    xx[1:]- xx[:-1],
                    yy[1:]- yy[:-1],
                    zz[1:]- zz[:-1],
            ]).transpose()])
            grad      = make_unit(grad)
            grad_mean = grad.mean(0)
            grad_var  = grad.var(0)
            #print(grad_mean)

            if (xx[1]<xx[2]) or (xx[0]<xx[1]) :
            #if np.any(grad_var>0.005):
            #if np.all(grad_var[0]<0.01):
                invalid_ids.append(p)
                continue

            # select only straight tracks for initial experiments ------------------

        for p in invalid_ids:
            particle_id[particle_id==p] *= -1

        data = pd.DataFrame(
            data = np.column_stack((x, y, z, hit_id, particle_id, layer_id)),
            columns=['x', 'y', 'z', 'hit_id', 'particle_id', 'layer_id'])



        #combinations (triplets) -------------------------------------------------------------
        d0 =  data.loc[data.layer_id==2]
        d1 =  data.loc[data.layer_id==4]
        d2 =  data.loc[data.layer_id==6]
        L0 = len(d0)
        L1 = len(d1)
        L2 = len(d2)
        print(L0,L1,L2)

        d0_xyz = d0.as_matrix(columns=['x', 'y', 'z', 'hit_id']).reshape(-1,4)
        d1_xyz = d1.as_matrix(columns=['x', 'y', 'z', 'hit_id']).reshape(-1,4)
        d2_xyz = d2.as_matrix(columns=['x', 'y', 'z', 'hit_id']).reshape(-1,4)
        d0_p   = d0.as_matrix(columns=['particle_id']).reshape(-1)
        d1_p   = d1.as_matrix(columns=['particle_id']).reshape(-1)
        d2_p   = d2.as_matrix(columns=['particle_id']).reshape(-1)

        d0_p   = np.repeat(d0_p,[L1*L2,])
        d1_p   = np.tile(np.repeat(d1_p,[L2,]),[L0,])
        d2_p   = np.tile(d2_p,[L0*L1,])
        d012_p = np.vstack([d0_p,d1_p,d2_p]).T

        d0_xyz = np.tile(d0_xyz,[1,L1*L2]).reshape(-1,4)
        d1_xyz = np.tile(np.tile(d1_xyz,[1,L2]).reshape(-1,4),[L0,1])
        d2_xyz = np.tile(d2_xyz,[L0*L1,1])
        d012_xyz = np.concatenate([d0_xyz,d1_xyz,d2_xyz],1).reshape(-1,3,4)

        combination = d012_xyz
        label = np.logical_and(
             np.all(d012_p==d012_p[:,0].reshape(-1,1),1),  #same label
            ~np.any(d012_p<=0,1)
        ).astype(np.uint8)

        zz=0
        if 1: #draw results
            ax.clear()
            set_figure(ax, title='event %s'%event)
            draw_hit( ax, h )


            for i in np.where(label==1)[0]:
                #clear_figure(ax)
                #draw_hit( ax, h )

                tracklet = combination[i]
                color=np.random.uniform(0,1,(3))
                ax.plot(tracklet[:,0],tracklet[:,1],tracklet[:,2],'.-',markersize=8,color=color)

                #plt.pause(0.01)

        ax.set_xlim(0,-100), ax.set_xlabel('x', fontsize=16)
        ax.set_ylim(-20,20), ax.set_ylabel('y', fontsize=16)
        ax.set_zlim(500,1000), ax.set_zlabel('z', fontsize=16)
        #plt.show()
        plt.pause(0.01)
        #input('Press any key to continue.')
        #exit(0)

        samples.append((event,data,combination,label))
        continue
        ##-----------------------------------------------------------------------------


    #pickle_file = '/root/share/project/kaggle/cern/data/samples_more.pickle'
    save_pickle_file(data_dir+'/user_data/samples_train_2.pickle', samples[:-2])
    save_pickle_file(data_dir+'/user_data/samples_valid_2.pickle', samples[-2:])
    zz=0



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_explore()

    print('\nsucess!')

