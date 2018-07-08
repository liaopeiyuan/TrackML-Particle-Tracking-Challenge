import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2' #'3,2,1,0'

from common import *
from utility.file import *
from net.rate   import *
from net.metric import *
from dataset.others import *

from model import LstmNet as Net


def train_augment(data, combination, label, index):

    tracklet = []
    truth    = []
    idx      = []

    #sample positive
    pos = np.where(label==1)[0]
    num_pos = len(pos)

    for i in pos:
        t = combination[i][:,0:3].reshape(1,9)
        tracklet.append(t)
        truth.append(1)

    #sample negative
    num_neg = 2 * num_pos
    neg = np.where(label==0)[0]
    idx = random.sample( range(0, len(neg)), num_neg)
    neg = neg[idx]

    for i in neg:
        t = combination[i][:,0:3].reshape(1,9)
        tracklet.append(t)
        truth.append(0)


    tracklet = np.vstack(tracklet)
    truth    = np.array(truth)
    tracklet = torch.from_numpy(tracklet).float()
    truth    = torch.from_numpy(truth).float()

    length = num_neg + num_pos
    return tracklet, truth, data, label, length, index


def train_collate(batch):
    batch_size = len(batch)
    i = 0

    tracklets  = torch.cat([batch[b][i]for b in range(batch_size)], 0); i=i+1
    truths     = torch.cat([batch[b][i]for b in range(batch_size)], 0); i=i+1

    datas      =           [batch[b][i]for b in range(batch_size)]; i=i+1
    labels     =           [batch[b][i]for b in range(batch_size)]; i=i+1
    lengths    =           [batch[b][i]for b in range(batch_size)]; i=i+1
    indices    =           [batch[b][i]for b in range(batch_size)]; i=i+1

    return [tracklets, truths,  datas, labels, lengths, indices]


def load_one_train_data(event_id,):

    data_dir  = '/mydisk/TrackML-Data'
    particles = pd.read_csv(data_dir + '/train_100_events/event%s-particles.csv'%event_id)
    hits  = pd.read_csv(data_dir + '/train_100_events/event%s-hits.csv' %event_id)
    truth = pd.read_csv(data_dir + '/train_100_events/event%s-truth.csv'%event_id)
    #cells = pd.read_csv(data_dir + '/train_100_events/event%s-cells.csv'%event_id)

    truth = truth.merge(hits,       on=['hit_id'],      how='left')
    truth = truth.merge(particles,  on=['particle_id'], how='left')

    #--------------------------------------------------------
    df = truth
    return (df)


def make_train_one_batch(df): #4096

    # ..........................................
    # use volume feature in "x,y,z > 0" only
    df = df.copy()
    df = df.loc[ (df.x>0) & (df.y>0) & (df.z>0) ]
    df = df.assign(r   = np.sqrt( df.x**2 + df.y**2))
    df = df.assign(d   = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
    df = df.assign(a   = np.arctan2(df.y, df.x))
    df = df.assign(cosa= np.cos(df.a))
    df = df.assign(sina= np.sin(df.a))
    df = df.assign(phi = np.arctan2(df.z, df.r))
    N  = len(df)

    a,r,z = df[['a', 'r', 'z' ]].values.astype(np.float32).T
    p = df['particle_id'].values.astype(np.int64)



    #print('N=%d'%N)
    particle_ids = list(df.particle_id.unique())
    num_particle_ids = len(particle_ids)

    input  = np.column_stack((r/1000,a,z/3000))
    tracks = []
    for particle_id in particle_ids:
        if particle_id==0: continue
        t = np.where(p==particle_id)[0]
        t = t[np.argsort(r[t])]

        if len(t)<10: continue
        track = input[t[:10]]
        tracks.append(track)

    tracks = np.array(tracks)
    input  = tracks
    truth  = tracks[:,:,:3]
    #<debug only>---------------------------------------------------------

    input  = torch.from_numpy(input).cuda()
    truth  = torch.from_numpy(truth).cuda()


    return (df, input, truth)

make_test_one_batch=make_train_one_batch






### training ##############################################################


# draw background ------------------
def draw_background(df,x,y,z,a,r,ax,ax1):
    ax.clear()
    ax1.clear()
    ax.plot(a,r,'.',color = [0.75,0.75,0.75], markersize=5)
    ax1.plot(x,y,z,'.',color = [0.75,0.75,0.75], markersize=5)
    particle_ids = list(df.particle_id.unique())
    num_particle_ids = len(particle_ids)
    p = df['particle_id'].values.astype(np.int64)

    for particle_id in particle_ids:
        if particle_id==0: continue
        t = np.where(p==particle_id)[0]
        t = t[np.argsort(z[t])]
        ax.plot(a[t],r[t],'.-',color = [0.75,0.75,0.75], markersize=5)
        ax1.plot(x[t],y[t],z[t],'.-',color = [0.75,0.75,0.75], markersize=5)



def run_train():

    out_dir = RESULTS_DIR + '/set53'
    initial_checkpoint = \
       None # RESULTS_DIR + '/set53/checkpoint/00008400_model.pth'

    pretrain_file =  None
    skip = [ ]

    ## setup  -----------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/train', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    fig = plt.figure(figsize=(7,7))
    ax  = fig.add_subplot(111)
    fig.patch.set_facecolor('white')

    fig1 = plt.figure(figsize=(8,8))
    ax1  = fig1.add_subplot(111, projection='3d')
    fig1.patch.set_facecolor('white')

    # fig2 = plt.figure(figsize=(5,5))
    # ax2  = fig2.add_subplot(111, projection='3d')



    ## net ----------------------
    log.write('** net setting **\n')
    net = Net(3,3).cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        # cfg = load_pickle_file(out_dir +'/checkpoint/configuration.pkl')

    if pretrain_file is not None:
        log.write('\tpretrain_file = %s\n' % pretrain_file)
        net.load_pretrain( pretrain_file, skip)

    log.write('%s\n\n'%(type(net)))
    log.write('\n')



    ## optimiser ----------------------------------
    iter_accum  = 1
    batch_size  = 1

    num_iters   = 1000  *1000
    iter_smooth = 20
    iter_log    = 50
    iter_valid  = 100
    iter_save   = [0, num_iters-1]\
                   + list(range(0,num_iters,100))#1*1000


    LR = None  #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.01/iter_accum, momentum=0.9, weight_decay=0.0001)

    start_iter = 0
    start_epoch= 0.
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']

        rate = get_learning_rate(optimizer)  #load all except learning rate
        optimizer.load_state_dict(checkpoint['optimizer'])
        adjust_learning_rate(optimizer, rate)
        pass


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    train_dataset = [] #<todo> replace this with a data iterator !!!!
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        # sampler = SequentialSampler(train_dataset),
        batch_size=batch_size,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_collate)

    event_ids = ['000001025',]  #000001029
    for event_id in event_ids:
        df = load_one_train_data(event_id)
        train_dataset.append(df)


    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' momentum=%f\n'% optimizer.param_groups[0]['momentum'])
    log.write(' LR=%s\n\n'%str(LR) )

    #log.write(' samples_per_epoch = %d\n\n'%len(train_dataset))
    log.write(' rate    iter   epoch  num   | valid_loss               | train_loss               | batch_loss               |  time          \n')
    log.write('-------------------------------------------------------------------------------------------------------------------------------\n')

    train_loss  = np.zeros(6,np.float32)
    valid_loss  = np.zeros(6,np.float32)
    batch_loss  = np.zeros(6,np.float32)
    rate = 0

    start = timer()
    j = 0
    i = 0

    counter =0
    while  i<num_iters:  # loop over the dataset multiple times
        sum_train_loss = np.zeros(6,np.float32)
        sum = 0


        optimizer.zero_grad()
        if 1: #<todo> replace this with a data iterator !!!!
            net.set_mode('train')

            df = train_dataset[0]
            df1,  input, truth = make_train_one_batch(df) #32


            #check data============================================

            # if 0:
            #     input = input.data.cpu().numpy()


            #======================================================



            #------------------------------------------------------------------------------------------
            len_train_dataset = 100000000  #len(train_dataset)
            batch_size = len(input)   #len(indices)
            i = j/iter_accum + start_iter
            epoch = (i-start_iter)*batch_size*iter_accum/len_train_dataset + start_epoch
            num_products = epoch*len_train_dataset

            if i % iter_valid==0:
                net.set_mode('valid')
                #valid_loss = evaluate(net, valid_loader)
                net.set_mode('train')

                print('\r',end='',flush=True)
                log.write('%0.4f %5.1f k %6.1f %4.1f m |  %0.3f  |  %0.3f  |  %0.5f  | %s\n' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], #valid_loss[1], valid_loss[2], valid_loss[3], #valid_loss[4], valid_loss[5],#valid_acc,
                         train_loss[0], #train_loss[1], train_loss[2], train_loss[3], #train_loss[4], train_loss[5],#train_acc,
                         batch_loss[0], #batch_loss[1], batch_loss[2], batch_loss[3], #batch_loss[4], batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60)))
                time.sleep(0.01)

            #if 1:
            if i in iter_save:
                torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(i))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter'     : i,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(i))

            # learning rate schduler -------------
            if LR is not None:
                lr = LR.get_rate(i)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr/iter_accum)
            rate = get_learning_rate(optimizer)*iter_accum


            # one iteration update  -------------
            estimate = net(input)
            loss  = net.criterion(estimate,truth)

            # accumulated update
            loss.backward()
            if j%iter_accum == 0:
                #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()


            # print statistics  ------------
            batch_loss = np.array((
                           loss.cpu().data.numpy(),
                           0,
                           0,
                           0,
                           0,
                           0,
                         ))
            sum_train_loss += batch_loss
            sum += 1
            if i%iter_smooth == 0:
                train_loss = sum_train_loss/sum
                sum_train_loss = np.zeros(6,np.float32)
                sum = 0


            print('\r%0.4f %5.1f k %6.1f %4.1f m |  %0.3f  |  %0.3f  |  %0.5f  | %s  %d,%d,%s' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], #valid_loss[1], valid_loss[2], valid_loss[3], #valid_loss[4], valid_loss[5],#valid_acc,
                         train_loss[0], #train_loss[1], train_loss[2], train_loss[3], #train_loss[4], train_loss[5],#train_acc,
                         batch_loss[0], #batch_loss[1], batch_loss[2], batch_loss[3], #batch_loss[4], batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60) ,i,j, ''), end='',flush=True)#str(inputs.size()))

            j=j+1

            #<debug> ===================================================================

            # if i%20==0 and i!=0:
            #     # if i%200==0:
            #     # print('show estimate ....')
            #     net.set_mode('test')
            #     with torch.no_grad():
            #         logit = net(input)
            #         prob = F.sigmoid(logit)
            #
            #     input = input.data.cpu().numpy()
            #     #query = input.query.cpu().numpy()
            #     prob = prob.data.cpu().numpy()
            #     #truth = truth.data.cpu().numpy()
            #
            #     ax1.clear()
            #     x, y, z = input[0].T
            #     ax1.plot(x, y, z, '.', color=[0.75, 0.75, 0.75], markersize=5, linewidth=0)
            #
            #     N = len(prob)
            #     for n in range(N):
            #         p = prob[n]
            #         idx = np.where(prob[n] > 0.6)[0]
            #
            #         if len(idx) > 0:
            #             color = np.random.uniform(0, 1, 3)
            #             ax1.plot(x[idx], y[idx], z[idx], '.', color=color, markersize=10, linewidth=0)
            #
            #             # ax1.plot([0,],[0,],[0,],'.',color = [0,0,0], markersize=10, linewidth=0)
            #
            #             plt.pause(0.01)
            #

            # for triplets, truths, logits, labels, lengths, indices in train_loader:
            #     x,y,z,a,r = input.T
            #     draw_background(df1,x,y,z,a,r,ax,ax1)
            #
            #     num_windows = (len(triplets))
            #
            #     triplet = triplets[n].data.cpu().numpy()
            #     truths = truths[n].data.cpu().numpy()
            #     logit = logits[n].data.cpu().numpy()
            #     prob  = np_sigmoid(logit)
            #
            #     split = np.cumsum(lengths)
            #     triplets = np.split(triplets,split)
            #     truths = np.split(truths, split)
            #
            #     for b in range(batch_size):
            #         ax.clear()
            #
            #         data = logits[b]
            #         x = data.x.values
            #         y = data.y.values
            #         z = data.z.values
            #
            #         ax.plot(x,y,z,'.',color=[])
            #          #draw estimate ---
            #          T = np.where(prob>0.8)[0]
            #          #T = np.where(truth>0.5)[0]
            #          for t in T:
            #              idx = triplet[t]
            #              ax.plot (a[idx],r[idx],'.-',color = [1,0,0], markersize=8)
            #              ax1.plot(x[idx],y[idx],z[idx],'.-',color = [1,0,0], markersize=8)
            #     # plt.waitforbuttonpress(5)
            #      plt.pause(0.01)
            #
            #     plt.show()


            net.set_mode('train')


    if 1: #save last
        torch.save(net.state_dict(),out_dir +'/checkpoint/%d_model.pth'%(i))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
            'epoch'    : epoch,
        }, out_dir +'/checkpoint/%d_optimizer.pth'%(i))

    log.write('\n')


# main #################################################################


def run_valid():

    out_dir = RESULTS_DIR + '/set53'
    initial_checkpoint = None
        #RESULTS_DIR + '/set53/checkpoint/00009800_model.pth'

    # out_dir = RESULTS_DIR + '/set11'
    # initial_checkpoint = \
    #     RESULTS_DIR + '/set11/checkpoint/00002200_model.pth'

    #start experiments here!
    os.makedirs(out_dir +'/backup', exist_ok=True)
    os.makedirs(out_dir +'/valid', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.evaluate.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    fig = plt.figure(figsize=(7,7))
    ax  = fig.add_subplot(111, )
    fig.patch.set_facecolor('white')

    fig1 = plt.figure(figsize=(8,8))
    ax1  = fig1.add_subplot(111, projection='3d')
    fig1.patch.set_facecolor('white')

    fig2 = plt.figure(figsize=(8,8))
    ax2  = fig2.add_subplot(111, projection='3d')
    fig2.patch.set_facecolor('white')

    ## net ------------------------------
    net = Net(3,3).cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n\n'%(type(net)))
    log.write('\n')



    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    event_ids = ['000001029',]  #'000001029',  000001025


    for event_id in event_ids:
        df = load_one_train_data(event_id)


        estimates=[]
        if 1:
            #<todo> cut the volume, etc
            df1, input, truths = make_test_one_batch(df)

            net.set_mode('test')
            with torch.no_grad():
                estimate = net.forward(input)

            #-------------------------------------------------
            input     = input.data.cpu().numpy()
            estimate  = estimate.data.cpu().numpy()

            batch_size = len(estimate)
            for n in range(0,batch_size,2):

                a,r,z = input[n].T
                x = r*np.cos(a)
                y = r*np.sin(a)

                ea,er,ez = estimate[n].T
                ex = er*np.cos(ea)
                ey = er*np.sin(ea)


                color = np.random.uniform(0,1,3)
                ax1.plot(ex,ey,ez,'.-',color = [0.75,0.75,0.75], markersize=10)
                ax1.plot(x,y,z,'.-',color = color, markersize=5)
                ax2.plot(ea,er,ez,'.-',color = [0.75,0.75,0.75], markersize=10)
                ax2.plot(a,r,z,'.-',color = color, markersize=5)
                if n==50: plt.show(1)

            #plt.waitforbuttonpress(5)
            #plt.pause(0.01)


            #plt.pause(0.01)
            plt.show()


    pass




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_train()
    run_train()


    print('\nsucess!')#



#  ffmpeg -f image2  -pattern_type glob -r 33 -i "iterations/*.png" -c:v libx264  iterations.mp4
#  convert *.png animated.gif
#