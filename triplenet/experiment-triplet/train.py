import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2' #'3,2,1,0'

from dataset.others import *

from reader import *
from model import TripletNet as Net

from common import *
from utility.file import *
from net.rate   import *
from net.metric import *



# -------------------------------------------------------------------------------------

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


### training ##############################################################
# def evaluate( net, test_loader ):
#
#     test_num  = 0
#     test_loss = np.zeros(6,np.float32)
#     return test_loss
#
#     for i, (inputs, foregrounds_truth, cuts_truth, images, masks_truth, indices) in enumerate(test_loader, 0):
#
#         with torch.no_grad():
#             inputs            = Variable(inputs).cuda()
#             foregrounds_truth = Variable(foregrounds_truth).cuda()
#             cuts_truth        = Variable(cuts_truth).cuda()
#
#             net.forward( inputs )
#             net.criterion( foregrounds_truth, cuts_truth )
#
#         batch_size = len(indices)
#         test_loss += batch_size*np.array((
#                            net.loss.cpu().data.numpy(),
#                            net.foreground_loss.cpu().data.numpy(),
#                            net.cut_loss.cpu().data.numpy(),
#                            0,
#                            0,
#                            0,
#                          ))
#         test_num  += batch_size
#
#     assert(test_num == len(test_loader.sampler))
#     test_loss = test_loss/test_num
#     return test_loss



#--------------------------------------------------------------
def run_train():

    out_dir = RESULTS_DIR + '/xx10'
    initial_checkpoint = None
        #RESULTS_DIR + '/xx10/checkpoint/00002200_model.pth'
        

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


    # fig = plt.figure(figsize=(5,5))
    # ax  = fig.add_subplot(111, projection='3d')

    fig1 = plt.figure(figsize=(5,5))
    ax1  = fig1.add_subplot(111, projection='3d')

    fig2 = plt.figure(figsize=(5,5))
    ax2  = fig2.add_subplot(111, projection='3d')



    ## net ----------------------
    log.write('** net setting **\n')
    net = Net().cuda()

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
    batch_size  = 8

    num_iters   = 1000  *1000
    iter_smooth = 20
    iter_log    = 50
    iter_valid  = 100
    iter_save   = [0, num_iters-1]\
                   + list(range(0,num_iters,200))#1*1000


    LR = None  #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.001/iter_accum, momentum=0.9, weight_decay=0.0001)

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

    train_dataset = DummyDataset(
                            'samples_train',
                            mode='<not_used>',transform = train_augment)

    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),
                        #sampler = SequentialSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = train_collate)



    # log.write('\ttrain_dataset.split = %s\n'%(train_dataset.split))
    # log.write('\tvalid_dataset.split = %s\n'%(valid_dataset.split))
    log.write('\tlen(train_dataset)  = %d\n'%(len(train_dataset)))
    # log.write('\tlen(valid_dataset)  = %d\n'%(len(valid_dataset)))
    # log.write('\tlen(train_loader)   = %d\n'%(len(train_loader)))
    # log.write('\tlen(valid_loader)   = %d\n'%(len(valid_loader)))
    log.write('\tbatch_size  = %d\n'%(batch_size))
    log.write('\titer_accum  = %d\n'%(iter_accum))
    log.write('\tbatch_size*iter_accum  = %d\n'%(batch_size*iter_accum))
    log.write('\n')

    #<debug>========================================================================================
    if 0:
        #fig = plt.figure(figsize=(5,5))
        #ax  = fig.add_subplot(111, projection='3d')

        for tracklets, truths, datas, labels, lengths, indices in train_loader:

            batch_size = len(indices)
            print('batch_size=%d'%batch_size)

            tracklets = tracklets.data.cpu().numpy()
            truths = truths.data.cpu().numpy()
            split  = np.cumsum(lengths)
            tracklets = np.split(tracklets,split)
            truths    = np.split(truths,split)

            for b in range(batch_size):
                ax.clear()

                data = datas[b]
                x = data.x.values
                y = data.y.values
                z = data.z.values
                ax.plot(x,y,z,'.',color = [0.75,0.75,0.75], markersize=3)

                tracklet = tracklets[b].reshape(-1,3,3)
                truth = truths[b]

                pos = np.where(truth==1)[0]
                for i in pos:
                    t = tracklet[i]
                    color = np.random.uniform(0,1,(3))
                    ax.plot(t[:,0],t[:,1],t[:,2],'.-',color = color, markersize=6)

                set_figure(ax,x_limit=(0,-100),y_limit=(-20,20),z_limit=(500,1000))
                plt.pause(0.01)
        plt.show()


    #<debug>========================================================================================


    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' momentum=%f\n'% optimizer.param_groups[0]['momentum'])
    log.write(' LR=%s\n\n'%str(LR) )

    log.write(' images_per_epoch = %d\n\n'%len(train_dataset))
    log.write(' rate    iter   epoch  num   | valid_loss        | train_loss       | batch_loss      |  time       \n')
    log.write('----------------------------------------------------------------------------------------------------\n')

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

        for tracklets, truths, datas, labels, lengths, indices in train_loader:

            batch_size = len(indices)
            i = j/iter_accum + start_iter
            epoch = (i-start_iter)*batch_size*iter_accum/len(train_dataset) + start_epoch
            num_products = epoch*len(train_dataset)

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

            # learning rate schduler -------------
            if LR is not None:
                lr = LR.get_rate(i)
                if lr<0 : break
                adjust_learning_rate(optimizer, lr/iter_accum)
            rate = get_learning_rate(optimizer)*iter_accum


            # one iteration update  -------------
            tracklets = tracklets.cuda()
            truths    = truths.cuda()

            logits = net.forward( tracklets )
            loss = F.binary_cross_entropy_with_logits(logits,truths)



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


            print('\r%0.4f %5.1f k %6.1f %4.1f m |  %0.3f  |  %0.3f  |  %0.3f  | %s  %d,%d,%s' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], #valid_loss[1], valid_loss[2], valid_loss[3], #valid_loss[4], valid_loss[5],#valid_acc,
                         train_loss[0], #train_loss[1], train_loss[2], train_loss[3], #train_loss[4], train_loss[5],#train_acc,
                         batch_loss[0], #batch_loss[1], batch_loss[2], batch_loss[3], #batch_loss[4], batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60) ,i,j, ''), end='',flush=True)#str(inputs.size()))
            j=j+1

            #<debug> ===================================================================
            if 1:
            #if i%200==0:
                net.set_mode('test')
                with torch.no_grad():
                    logits  = net.forward( tracklets )

                tracklets  = tracklets.data.cpu().numpy()
                probs  = np_sigmoid(logits.data.cpu().numpy())
                truths = truths.data.cpu().numpy()

                batch_size = len(indices)
                split     = np.cumsum(lengths)
                tracklets = np.split(tracklets,split)
                probs     = np.split(probs,split)
                truths    = np.split(truths,split)

                for b in range(batch_size):
                    ax1.clear()
                    ax2.clear()

                    data = datas[b]
                    x = data.x.values
                    y = data.y.values
                    z = data.z.values
                    ax1.plot(x,y,z,'.',color = [0.75,0.75,0.75], markersize=3)
                    ax2.plot(x,y,z,'.',color = [0.75,0.75,0.75], markersize=3)

                    tracklet = tracklets[b]
                    prob     = probs[b]
                    truth    = truths[b]

                    #idx = np.where(prob>0.5)[0]
                    #for i in idx:
                    threshold=0.5
                    for i in range(len(truth)):
                        t = tracklet[i].reshape(-1,3)

                        if prob[i]>threshold and truth[i]>0.5:  #hit
                            color=np.random.uniform(0,1,(3))
                            ax1.plot(t[:,0],t[:,1],t[:,2],'.-',color = color, markersize=6)

                        if prob[i]>threshold and truth[i]<0.5:  #fp
                            ax2.plot(t[:,0],t[:,1],t[:,2],'.-',color = [0,0,0], markersize=6)
                        if prob[i]<threshold and truth[i]>0.5:  #miss
                            ax2.plot(t[:,0],t[:,1],t[:,2],'.-',color = [1,0,0], markersize=6)

                    set_figure(ax1, title='hit   @sample%d'%indices[b],x_limit=(0,-100),y_limit=(-20,20),z_limit=(500,1000))
                    set_figure(ax2, title='error @sample%d'%indices[b],x_limit=(0,-100),y_limit=(-20,20),z_limit=(500,1000))
                    plt.pause(0.01)
                    #fig.savefig(out_dir +'/train/%05d.png'%indices[b])
                pass


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


# main #################################################################


def run_predict():

    out_dir = RESULTS_DIR + '/xx10'
    initial_checkpoint = \
        RESULTS_DIR + '/xx10/checkpoint/00002200_model.pth'
        #None #


    #start experiments here!
    os.makedirs(out_dir +'/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.evaluate.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    fig1 = plt.figure(figsize=(5,5))
    ax1  = fig1.add_subplot(111, projection='3d')
    fig2 = plt.figure(figsize=(5,5))
    ax2  = fig2.add_subplot(111, projection='3d')

    ## net ------------------------------
    net = Net().cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n\n'%(type(net)))
    log.write('\n')



    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    dataset = DummyDataset(
        'samples_valid',  #'samples_train',
        mode='<not_used>',transform = None,
    )

    log.write('\n')


    for n in range(len(dataset)):

        data, combination, label, index = dataset[n]
        num_combination = len(combination)
        estimate = np.zeros(num_combination,np.float32)

        if 1:
            ax1.clear()
            ax2.clear()
            x = data.x.values
            y = data.y.values
            z = data.z.values
            ax1.plot(x,y,z,'.',color = [0.75,0.75,0.75], markersize=3)
            ax2.plot(x,y,z,'.',color = [0.75,0.75,0.75], markersize=3)




        for j in range(0, num_combination, 512):
            batch_size = min(j+512,num_combination)- j

            truth     = label[j:j+batch_size]
            tracklets = combination[j:j+batch_size,:,0:3].reshape(-1,9)
            tracklets = np.vstack(tracklets)
            tracklets = torch.from_numpy(tracklets).float()

            net.set_mode('test')
            with torch.no_grad():
                tracklets = tracklets.cuda()
                logits = net.forward( tracklets )

            tracklet = tracklets.data.cpu().numpy()
            prob = np_sigmoid(logits.data.cpu().numpy())
            estimate[j:j+batch_size] = prob


            #draw results -------------------------------------------------------------
            if 1:
                threshold = 0.5
                for i in range(batch_size):
                    #print(j,i)
                    t = tracklet[i].reshape(-1,3)

                    if prob[i]>threshold and truth[i]>0.5:  #hit
                        print('*',end='',flush=True)
                        color=np.random.uniform(0,1,(3))
                        ax1.plot(t[:,0],t[:,1],t[:,2],'.-',color = color, markersize=6)

                    if prob[i]>threshold and truth[i]<0.5:  #fp
                        ax2.plot(t[:,0],t[:,1],t[:,2],'.-',color = [0,0,0], markersize=6)
                    if prob[i]<threshold and truth[i]>0.5:  #miss
                        ax2.plot(t[:,0],t[:,1],t[:,2],'.-',color = [1,0,0], markersize=6)

                    set_figure(ax1, title='hit   @sample%d'%index,x_limit=(0,-100),y_limit=(-20,20),z_limit=(500,1000))
                    set_figure(ax2, title='error @sample%d'%index,x_limit=(0,-100),y_limit=(-20,20),z_limit=(500,1000))
                    #plt.pause(0.01)

        estimate=(estimate>0.5)
        label   =(label>0.5)
        print('\n@sample%d :'%index)
        print('\t total  = %d '%(len(label)))
        print('\t hit  = %d (%0.2f)'%((estimate*label).sum(), (estimate*label).sum()/label.sum()))
        print('\t miss = %d (%0.2f)'%((~estimate*label).sum(), (~estimate*label).sum()/label.sum()))
        print('\t fp   = %d (%0.2f)'%((estimate*(~label)).sum(), (estimate*(~label)).sum()/estimate.sum()))

        #plt.show()
        plt.pause(5)
    plt.show()
    pass




    #assert(test_num == len(test_loader.sampler))
    log.write('-------------\n')
    log.write('initial_checkpoint  = %s\n'%(initial_checkpoint))
    #log.write('tag=%s\n'%tag)
    log.write('\n')






# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()
    run_predict()


    print('\nsucess!')



#  ffmpeg -f image2  -pattern_type glob -r 33 -i "iterations/*.png" -c:v libx264  iterations.mp4
#
#
