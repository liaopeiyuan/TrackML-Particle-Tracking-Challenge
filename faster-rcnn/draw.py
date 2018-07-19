from common import *

FIG1 = plt.figure(figsize=(8,8))
AX1  = FIG1.add_subplot(111, )
FIG1.patch.set_facecolor('white')
FIG2 = plt.figure(figsize=(8,8))
AX2  = FIG2.add_subplot(111, )
FIG2.patch.set_facecolor('white')


FIG3d1 = plt.figure(figsize=(8,8))
AX3d1  = FIG3d1.add_subplot(111, projection='3d')
FIG3d1.patch.set_facecolor('white')
FIG3d2 = plt.figure(figsize=(8,8))
AX3d2  = FIG3d2.add_subplot(111, projection='3d')
FIG3d2.patch.set_facecolor('white')




def plot3d_particles(ax3d, particle_ids, p, ar,r,zr, z,min_length=3, color=[0.75,0.75,0.75], markersize=0,  linewidth=1, subsample=1):

    num_particle_ids = len(particle_ids)
    for n in range(0,num_particle_ids,subsample):
        particle_id = particle_ids[n]
        t = np.where(p==particle_id)[0]
        if len(t)<min_length: continue

        #skip angle discontinous tracks
        if np.fabs(ar[t[0]] - ar[t[-1]])>1: continue
        t = t[np.argsort(np.fabs(z[t]))]

        ax3d.plot(ar[t], r[t], zr[t],'.-',  color=color, markersize=markersize,  linewidth=linewidth)

        #ax3d.plot(ar[t], r[t], zr[t],'.-',  color=np.random.uniform(0,1,3), markersize=16,  linewidth=1)
        #ax3d.plot(ar[t], r[t], zr[t],'.',   color=[0,0,0], markersize=16,  linewidth=8, mfc ='none')
        #ax3d.plot(ar[t], r[t], zr[t],'.-',  color=[0,0,0], markersize=0,  linewidth=8)
        #ax3d.plot(ar[[t[0],t[-1]]], r[[t[0],t[-1]]], zr[[t[0],t[-1]]],'-',  color=[0,0,0], markersize=0,  linewidth=8)
        #plt.waitforbuttonpress(-1)

def plot_particles(ax, particle_ids, p, ar, zr, z,min_length=3, color=[0.75,0.75,0.75], markersize=0,  linewidth=1, subsample=1):
    num_particle_ids = len(particle_ids)
    for n in range(0,num_particle_ids,subsample):
        particle_id = particle_ids[n]
        t = np.where(p==particle_id)[0]
        if len(t)<min_length: continue

        #skip angle discontinous tracks
        if np.fabs(ar[t[0]] - ar[t[-1]])>1: continue
        t = t[np.argsort(np.fabs(z[t]))]

        ax.plot(ar[t], zr[t],'.-',  color=color, markersize=markersize,  linewidth=linewidth)
        #ax.plot(ar[t], zr[t],'.-',  color=np.random.uniform(0,1,3), markersize=16,  linewidth=1)



#https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


def draw_voxel(ax3d, voxels, colors = None,markersize=3):
   v = voxels>0
   K = len(v)

   if colors is None: colors = plt.cm.gnuplot(np.arange(K)/K)

   for k in range(K):
        (j,i) = np.where(v[k])
        ax3d.plot(i,j,k,'.', color=colors[k],markersize=markersize)#edgecolors=


def plot_hits (ax, particle_ids, p, ar, zr, z,min_length=3, color=[0.75,0.75,0.75], markersize=0,  linewidth=1, subsample=1):
    num_particle_ids = len(particle_ids)
    for n in range(0,num_particle_ids,subsample):
        particle_id = particle_ids[n]
        t = np.where(p==particle_id)[0]
        if len(t)<min_length: continue

        #skip angle discontinous tracks
        if np.fabs(ar[t[0]] - ar[t[-1]])>1: continue
        t = t[np.argsort(np.fabs(z[t]))]

        ax.plot(ar[t], zr[t],'.-',  color=color, markersize=markersize,  linewidth=linewidth)
        #ax.plot(ar[t], zr[t],'.-',  color=np.random.uniform(0,1,3), markersize=16,  linewidth=1)

