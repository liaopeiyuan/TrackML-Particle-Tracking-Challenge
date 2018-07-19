import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2' #'3,2,1,0'

from common import *
from draw import *

from utility.file import *
from net.rate   import *
from net.metric import *
from dataset.others import *

from model import PairNet as Net


##--------------------------------------------------------

def point_to_data(
    a, zr, z, layer_id, p,
    a_limit, zr_limit,
    D,H,W, dist =0.10, subsample =-1
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


    # build neighbourhood graph
    pair = []
    link = []
    for d in range(D-1):
        l0 = np.where(layer_id==d  )[0]
        l1 = np.where(layer_id==d+1)[0]
        if (len(l0)==0) | (len(l1)==0) : continue


        q1  = np.column_stack((a[l1],zr[l1]))
        q0  = np.column_stack((a[l0],zr[l0]))
        kdt = KDTree(q1)  #<todo> fix angular discontinuinty
        n   = kdt.query_radius(q0, dist) #2

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

    if subsample>0:
        num = len(link)
        t    = np.random.choice(num, subsample)
        pair = pair[t]
        link = link[t]

    #if (len(pair)==0):
    #    print(a_limit,zr_limit)

    #----
    if (len(pair)==0):
        pair    = np.zeros((0,2),np.int32)
        location = np.zeros((8,0),np.int32)
        link     = np.zeros((0),np.float32)

    else:
        num= len(pair)

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

    return image, pair, location, link



##--------------------------------------------------------

def draw_data(
    a, zr, z, layer_id, p,
    image, pair, location, link, prob=None
):

    if prob is not None:
        prob = prob.reshape(-1)


    if 1:
        u = np.unique(pair.reshape(-1))
        depth = layer_id.max()

        AX3d1.clear()
        AX3d1.set_aspect('equal')
        AX3d1.scatter(a[u], zr[u], z[u], c=plt.cm.gnuplot( layer_id[u]/(depth+1) ), s=32, edgecolors='black') #edgecolors='none'

        L = len(link)
        for i in range(L):
            i0, i1 = pair[i]

            if 1:
                if link[i]>0.5:
                    AX3d1.plot(a[[i0,i1]], zr[[i0,i1]], z[[i0,i1]],'-', color=[0,0,0], markersize=5)

            if prob is not None:
                if prob[i]>0.5:
                    c = prob[i]
                    AX3d1.plot(a[[i0,i1]], zr[[i0,i1]], z[[i0,i1]],'-', color=[c,0,0],markersize=1)


        #set_axes_equal(AX3d1)
        AX3d1.set_title('points prob')



    if 1:
        C,H,W = image.shape

        AX3d2.clear()
        AX3d2.set_aspect('equal')
        draw_voxel(AX3d2, image>0.5, colors=np.zeros((C,3)), markersize=8)

        L = len(link)
        for i in range(L):
            k0, j0, i0, k1, j1, i1  = location[i]

            if 1:
                if link[i]>0.5:
                    AX3d2.plot([i0,i1],[j0,j1],[k0,k1],'-', color=[0,0,0],markersize=16)

            if prob is not None:
                if prob[i]>0.5:
                    c = prob[i]
                    AX3d2.plot([i0,i1],[j0,j1],[k0,k1],'-', color=[c,0,0],markersize=1)

        #set_axes_equal(AX3d2)
        AX3d2.set_title('voxel prob')




### training ##############################################################

def run_train():

    out_dir = RESULTS_DIR + '/seglink2.vox.5d-20x2'
    initial_checkpoint = \
         None  #RESULTS_DIR + '/seglink2.vox.5d-17/checkpoint/00001600_model.pth'


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


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')


    # =======================================================================
    train_dataset = [] #<todo> replace this with a data iterator !!!!
    event_ids = ['000001029','000001028','000001027','000001026','000001025']  #'000001030' #'000001025' #
    #event_ids = ['%09d'%i for i in range(1000,1100)]
    # event_ids=['000001030', '000001001']

    for event_id in event_ids:
        data_dir  = '/home/alexanderliao/data/Kaggle/competitions/trackml-particle-identification/results'
        #data_dir  = '/root/share/project/kaggle/cern/data/__download__/train_100_events'
        data = load_pickle_file(data_dir+'/data_1600.%s.3dcnn.npy'%event_id)
        a, zr, z, my_layer_id, p = data

        a_min  = a.min()
        a_max  = a.max()
        zr_min = zr.min()
        zr_max = zr.max()
        train_dataset.append([(a, zr, z, my_layer_id, p),(a_min,a_max,zr_min,zr_max)])

    # =======================================================================

    C,height,width =  8, 3200, 1600
    depth = 8
    H,W   = 256, 256




    ## net ----------------------
    log.write('** net setting **\n')
    net = Net(C, depth).cuda()


    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

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
                   + list(range(0,num_iters,200))#1*1000


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

        net.set_mode('train')
        optimizer.zero_grad()

        if 1: #<todo> replace this with a data iterator !!!!

            # make one batch ===============================================
            data =[]
            input=[]
            truth=[]
            idx0 =[]
            idx1 =[]
            pairwise =[]

            num_idx = -1 #1024
            batch_size = 4

            b=0
            while b <batch_size:
                (a, zr, z, my_layer_id, p),(a_min,a_max,zr_min,zr_max) =  train_dataset[np.random.choice(len(train_dataset))]

                # scale_a  =  6.4/1600
                # scale_zr = 12.8/3200
                # dist = 0.10

                scale_a  =  6.4/1600*2
                scale_zr = 12.8/3200*2
                dist = 0.10*2


                a0  = np.random.uniform( a_min, a_max-scale_a *W)  # -1.2 #
                zr0 = np.random.uniform(zr_min,zr_max-scale_zr*H)  #  4.5 #
                a1  = a0  + scale_a *W
                zr1 = zr0 + scale_zr*H

                image, pair, location, link = point_to_data( a, zr, z, my_layer_id, p, (a0,a1), (zr0,zr1), depth,H,W, dist=dist, subsample=num_idx )
                #draw_data( a, zr, z, my_layer_id, p, image, pair, location, link)
                #plt.show()

                if len(link)>0:
                    k0,j0,i0,  k1,j1,i1 = location.T

                    num = len(k0)
                    i0  = np.column_stack((np.full(num,b),k0,j0,i0,np.full(num,0)))
                    i1  = np.column_stack((np.full(num,b),k1,j1,i1,np.full(num,1)))

                    idx0.append(i0)
                    idx1.append(i1)
                    truth.append(link)
                    input.append(image)
                    pairwise.append(pair)
                    data.append((a, zr, z, my_layer_id, p))
                    b += 1

            idx0  = np.concatenate(idx0)
            idx1  = np.concatenate(idx1)
            pairwise = np.concatenate(pairwise)
            truth = np.concatenate(truth).astype(np.float32)
            input = np.array(input).astype(np.float32)

            #---

            idx0  = [ torch.from_numpy(idx0[:,i]).long().cuda() for i in range(5)]
            idx1  = [ torch.from_numpy(idx1[:,i]).long().cuda() for i in range(5)]
            input = torch.from_numpy(input).cuda()
            truth = torch.from_numpy(truth).cuda()

            # make one batch ===============================================



            #------------------------------------------------------
            len_train_dataset = 100000000  #len(train_dataset)
            batch_size = len(input)        #len(indices)
            i = j/iter_accum + start_iter
            epoch = (i-start_iter)*batch_size*iter_accum/len_train_dataset + start_epoch
            num_products = epoch*len_train_dataset

            if i % iter_valid==0:
                net.set_mode('valid')
                #valid_loss = evaluate(net, valid_loader)
                net.set_mode('train')

                print('\r',end='',flush=True)
                log.write('%0.4f %5.1f k %6.1f %4.1f m |  %0.3f  |  %0.3f  |  %0.3f  | %s\n' % (\
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
                pass

            # learning rate schduler -------------
            if LR is not None:
                lr = LR.get_rate(i)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr/iter_accum)
            rate = get_learning_rate(optimizer)*iter_accum


            # one iteration update  -------------
            logit = net( input, idx0, idx1 )
            loss = net.criterion(logit, truth)

            # accumulated update
            loss.backward()
            if j%iter_accum == 0:
                #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                pass


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


            print('\r%0.4f %5.1f k %6.1f %4.1f m |  %0.3f  |  %0.3f  |  %0.3f  | %s  %d,%d,%s' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], #valid_loss[1], valid_loss[2], valid_loss[3], #valid_loss[4], valid_loss[5],#valid_acc,
                         train_loss[0], #train_loss[1], train_loss[2], train_loss[3], #train_loss[4], train_loss[5],#train_acc,
                         batch_loss[0], #batch_loss[1], batch_loss[2], batch_loss[3], #batch_loss[4], batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60) ,i,j, ''), end='',flush=True)#str(inputs.size()))
            j=j+1

            #<debug> ===================================================================
            if 0:
            #if i%100==0:
                net.set_mode('test')
                with torch.no_grad():
                    logit = net( input, idx0, idx1 )
                    prob  = F.sigmoid(logit)

                #show only b in batch ---
                b=2

                input = input.data.cpu().numpy()
                truth = truth.data.cpu().numpy()
                prob = prob. data.cpu().numpy()
                idx0 = [idx0[i].data.cpu().numpy() for i in range(5)]
                idx1 = [idx1[i].data.cpu().numpy() for i in range(5)]
                idx0 = np.column_stack(idx0)
                idx1 = np.column_stack(idx1)

                ib = np.where(idx0[:,0]==b)[0] #first batch
                pair  = pairwise[ib]
                idx0  = idx0 [ib]
                idx1  = idx1 [ib]
                prob  = prob [ib]
                truth = truth[ib]

                input = input[b][:depth]
                a, zr, z, my_layer_id, p = data[b]

                location = np.column_stack((idx0[:,1],idx0[:,2],idx0[:,3],   idx1[:,1],idx1[:,2],idx1[:,3],))
                draw_data(a, zr, z, my_layer_id, p, input, pair, location, truth, prob)


                plt.pause(0.01)
                #plt.waitforbuttonpress(-1)
                plt.show()

                net.set_mode('train')
            #<debug> ===================================================================


        pass  #-- end of one data loader --
    pass #-- end of all iterations --


    if 1: #save last
        torch.save(net.state_dict(),out_dir +'/checkpoint/%d_model.pth'%(i))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
            'epoch'    : epoch,
        }, out_dir +'/checkpoint/%d_optimizer.pth'%(i))

    log.write('\n')





def run_test():

    out_dir = RESULTS_DIR + '/seglink2.vox.5d-20x2'
    initial_checkpoint = \
         RESULTS_DIR + '/seglink2.vox.5d-20x2/checkpoint/00005800_model.pth'



    ## setup  -----------------
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/test', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    #backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.test.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')


    # =======================================================================
    test_dataset = [] #<todo> replace this with a data iterator !!!!
    # event_ids = ['000001029','000001028','000001027','000001026','000001025']  #'000001030' #'000001025' #
    # event_id = '000001001''000001030'
    event_ids=['000001001']

    for event_id in event_ids:
        data_dir  = '/home/alexanderliao/data/Kaggle/competitions/trackml-particle-identification/results'
        #data_dir  = '/root/share/project/kaggle/cern/data/__download__/train_100_events'
        data = load_pickle_file(data_dir+'/data_1600.%s.3dcnn.npy'%event_id)
        a, zr, z, my_layer_id, p = data

        a_min  = a.min()
        a_max  = a.max()
        zr_min = zr.min()
        zr_max = zr.max()
        test_dataset.append([(a, zr, z, my_layer_id, p),(a_min,a_max,zr_min,zr_max)])

    # =======================================================================

    C,height,width =  8, 3200, 1600
    depth = 8
    H,W   = 512, 512




    ## net ----------------------
    log.write('** net setting **\n')
    net = Net(C, depth).cuda()


    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n\n'%(type(net)))
    log.write('\n')



    ## optimiser ----------------------------------


    while 1:
        if 1: #<todo> replace this with a data iterator !!!!

            # make one batch ===============================================
            data =[]
            input=[]
            truth=[]
            idx0 =[]
            idx1 =[]
            pairwise =[]

            num_idx = -1 #1024
            batch_size = 1

            b=0
            while b <batch_size:
                (a, zr, z, my_layer_id, p),(a_min,a_max,zr_min,zr_max) =  test_dataset[np.random.choice(len(test_dataset))]

                # scale_a  =  6.4/1600
                # scale_zr = 12.8/3200
                # dist = 0.10

                scale_a  =  6.4/1600*2
                scale_zr = 12.8/3200*2
                dist = 0.075*2


                a0  = np.random.uniform( a_min, a_max-scale_a *W)  # -1.2 #
                zr0 = np.random.uniform(zr_min,zr_max-scale_zr*H)  #  4.5 #
                a1  = a0  + scale_a *W
                zr1 = zr0 + scale_zr*H

                image, pair, location, link = point_to_data( a, zr, z, my_layer_id, p, (a0,a1), (zr0,zr1), depth,H,W, dist=dist, subsample=num_idx )
                #draw_data( a, zr, z, my_layer_id, p, image, pair, location, link)
                #plt.show()

                if len(link)>0:
                    k0,j0,i0,  k1,j1,i1 = location.T

                    num = len(k0)
                    i0 = np.column_stack((np.full(num,b),k0,j0,i0,np.full(num,0)))
                    i1 = np.column_stack((np.full(num,b),k1,j1,i1,np.full(num,1)))

                    idx0.append(i0)
                    idx1.append(i1)
                    truth.append(link)
                    input.append(image)
                    pairwise.append(pair)
                    data.append((a, zr, z, my_layer_id, p))
                    b += 1

            idx0  = np.concatenate(idx0)
            idx1  = np.concatenate(idx1)
            pairwise = np.concatenate(pairwise)
            truth = np.concatenate(truth).astype(np.float32)
            input = np.array(input).astype(np.float32)

            #---

            idx0  = [ torch.from_numpy(idx0[:,i]).long().cuda() for i in range(5)]
            idx1  = [ torch.from_numpy(idx1[:,i]).long().cuda() for i in range(5)]
            input = torch.from_numpy(input).cuda()
            truth = torch.from_numpy(truth).cuda()

            # make one batch ===============================================

            if 1:
            #if i%100==0:
                net.set_mode('test')
                with torch.no_grad():
                    logit = net( input, idx0, idx1 )
                    prob  = F.sigmoid(logit)

                #show only b in batch ---
                b=0

                input = input.data.cpu().numpy()
                truth = truth.data.cpu().numpy()
                prob = prob. data.cpu().numpy()
                idx0 = [idx0[i].data.cpu().numpy() for i in range(5)]
                idx1 = [idx1[i].data.cpu().numpy() for i in range(5)]
                idx0 = np.column_stack(idx0)
                idx1 = np.column_stack(idx1)

                ib = np.where(idx0[:,0]==b)[0] #first batch
                pair  = pairwise[ib]
                idx0  = idx0 [ib]
                idx1  = idx1 [ib]
                prob  = prob [ib]
                truth = truth[ib]

                input = input[b][:depth]
                a, zr, z, my_layer_id, p = data[b]

                location = np.column_stack((idx0[:,1],idx0[:,2],idx0[:,3],   idx1[:,1],idx1[:,2],idx1[:,3],))
                draw_data(a, zr, z, my_layer_id, p, input, pair, location, truth, prob)


                plt.pause(0.01)
                #plt.waitforbuttonpress(-1)
                plt.show()

                net.set_mode('train')
            #<debug> ===================================================================


        pass  #-- end of one data loader --
    pass #-- end of all iterations --




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()
    run_test()


    print('\nsucess!')



#  ffmpeg -f image2  -pattern_type glob -r 33 -i "iterations/*.png" -c:v libx264  iterations.mp4
#  convert *.png animated.gif
#
