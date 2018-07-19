from common import *
from draw import *


# 3d-cnn https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py
#        https://github.com/huangzhii/FCN-3D-pytorch
#        https://github.com/whitesnowdrop/c3d_pytorch

## block ## -------------------------------------

def make_2dconv_bn_relu(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):

    # non-square kernels and unequal stride and with padding
    # m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))

    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]



def make_linear_bn_relu(in_channels, out_channels):
    return [
        nn.Linear(in_channels, out_channels, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    ]



## net  ## -------------------------------------
class FeatureNet(nn.Module):

    def __init__(self, C=5, num_feature=64):
        super(FeatureNet, self).__init__()

        self.down1 = nn.Sequential(
            *make_2dconv_bn_relu(  C,  32, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu( 32,  32, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu( 32,  64, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
        )
        self.down2 = nn.Sequential(
            *make_2dconv_bn_relu( 64,  64, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu( 64,  64, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu( 64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
        )
        self.down3 = nn.Sequential(
            *make_2dconv_bn_relu(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
        )
        self.down4 = nn.Sequential(
            *make_2dconv_bn_relu(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
        )

        self.same = nn.Sequential(
            *make_2dconv_bn_relu(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
        )

        self.up4 = nn.Sequential(
            *make_2dconv_bn_relu(1024,256, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu( 256,256, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu( 256,256, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            #nn.Dropout(p=0.10),
        )
        self.up3 = nn.Sequential(
            *make_2dconv_bn_relu( 512,256, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu( 256,256, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu( 256,128, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
        )
        self.up2 = nn.Sequential(
            *make_2dconv_bn_relu( 256,128, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu( 128,128, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu( 128, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
        )
        self.up1 = nn.Sequential(
            *make_2dconv_bn_relu( 128, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu(  64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
            *make_2dconv_bn_relu(  64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1) ),
        )

        self.feature = nn.Sequential(
            nn.Conv2d(64, num_feature, kernel_size=(1,1), stride=(1,1), padding=(0,0) ),
        )



    def forward(self, v):

        down1 = self.down1(v)
        f,i1  = F.max_pool2d(down1, kernel_size=(2,2), stride=(2,2),return_indices=True)
        down2 = self.down2(f)
        f,i2  = F.max_pool2d(down2, kernel_size=(2,2), stride=(2,2),return_indices=True)
        down3 = self.down3(f)
        f,i3  = F.max_pool2d(down3, kernel_size=(2,2), stride=(2,2),return_indices=True)
        down4 = self.down4(f)
        f,i4  = F.max_pool2d(down4, kernel_size=(2,2), stride=(2,2),return_indices=True)

        f  = self.same(f)

        #out   = F.upsample(out, scale_factor=(2,2), mode='bilinear', align_corners=True) #12
        f = F.max_unpool2d(f, i4, kernel_size=(2,2), stride=(2,2))
        f = torch.cat([down4, f],1)
        f = self.up4(f)
        f = F.max_unpool2d(f, i3, kernel_size=(2,2), stride=(2,2))
        f = torch.cat([down3, f],1)
        f = self.up3(f)
        f = F.max_unpool2d(f, i2, kernel_size=(2,2), stride=(2,2))
        f = torch.cat([down2, f],1)
        f = self.up2(f)
        f = F.max_unpool2d(f, i1, kernel_size=(2,2), stride=(2,2))
        f = torch.cat([down1, f],1)
        f = self.up1(f)

        f = self.feature(f)

        return f


class PairNet(nn.Module):


    def __init__(self, C=5, depth=16, ):
        super(PairNet, self).__init__()
        self.depth = depth

        self.feature_net = FeatureNet(C,depth*2*32)
        self.logit = nn.Sequential(
            *make_linear_bn_relu(2*32, 512),
            *make_linear_bn_relu( 512, 256),
            *make_linear_bn_relu( 256,  64),
            nn.Linear( 64,  1),
        )


    def forward(self, v, idx0, idx1):

        f = data_parallel(self.feature_net, v)
        #f = self.feature_net(v)

        batch_size, C, H, W = f.shape
        f = f.permute(0,2,3,1).view(batch_size, H, W, self.depth, 2, -1)
        f0 = f[idx0[0],idx0[2],idx0[3],idx0[1],idx0[4]]
        f1 = f[idx1[0],idx1[2],idx1[3],idx1[1],idx1[4]]

        pair = torch.cat([f0, f1],1)
        logit = self.logit(pair)
        return logit



    def criterion(self, logit, truth):
        batch_size = logit.size(1)

        logit_flat = logit.contiguous().view (-1)
        truth_flat = truth.contiguous().view(-1)

        num     = len(truth_flat)
        num_pos = truth_flat.sum().item()+EPS
        num_neg = num-num_pos
        weight_flat = torch.cuda.FloatTensor(num).fill_(1)
        weight_flat[truth_flat>0.5]=0.1/num_pos
        weight_flat[truth_flat<0.5]=0.9/num_neg

        loss = F.binary_cross_entropy_with_logits(logit_flat,truth_flat,weight_flat,size_average=False)
        loss = loss.sum()/weight_flat.sum()
        return loss



    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


### run ##############################################################################




def run_check_net():

    batch_size  = 8
    C,H,W = 8, 256, 256
    depth = 8
    num_idx = 256

    input = np.random.uniform(-1,1, (batch_size,C,H,W )).astype(np.float32)
    truth = np.random.choice(2, (batch_size*num_idx)).astype(np.float32)

    idx0_b = []
    idx0_k = []
    idx0_j = []
    idx0_i = []
    idx0_t = []

    idx1_b = []
    idx1_k = []
    idx1_j = []
    idx1_i = []
    idx1_t = []

    for b in range(batch_size):
        i0_b = np.full(num_idx, b)
        i0_k = np.random.choice(depth, num_idx)
        i0_j = np.random.choice(H, num_idx)
        i0_i = np.random.choice(W, num_idx)
        i0_t = np.full(num_idx,0)
        idx0_b.append(i0_b)
        idx0_k.append(i0_k)
        idx0_j.append(i0_j)
        idx0_i.append(i0_i)
        idx0_t.append(i0_t)

        i1_b = np.full(num_idx, b)
        i1_k = np.random.choice(depth, num_idx)
        i1_j = np.random.choice(H, num_idx)
        i1_i = np.random.choice(W, num_idx)
        i1_t = np.full(num_idx,1)
        idx1_b.append(i1_b)
        idx1_k.append(i1_k)
        idx1_j.append(i1_j)
        idx1_i.append(i1_i)
        idx1_t.append(i1_t)



    idx0_b = np.concatenate(idx0_b)
    idx0_k = np.concatenate(idx0_k)
    idx0_j = np.concatenate(idx0_j)
    idx0_i = np.concatenate(idx0_i)
    idx0_t = np.concatenate(idx0_t)

    idx1_b = np.concatenate(idx1_b)
    idx1_k = np.concatenate(idx1_k)
    idx1_j = np.concatenate(idx1_j)
    idx1_i = np.concatenate(idx1_i)
    idx1_t = np.concatenate(idx1_t)

    idx0 = [idx0_b,idx0_k,idx0_j,idx0_i,idx0_t]
    idx1 = [idx1_b,idx1_k,idx1_j,idx1_i,idx1_t]



    idx0 = [torch.from_numpy(idx0[i]).long().cuda() for i in range(5)]
    idx1 = [torch.from_numpy(idx1[i]).long().cuda() for i in range(5)]
    input = torch.from_numpy(input).cuda()


    truth = torch.from_numpy(truth).cuda()

    #---

    net = PairNet( C=C, depth=depth ).cuda()
    net.set_mode('train')
    logit = net(input, idx0, idx1)
    loss  = net.criterion(logit, truth)
    print('loss:',loss)







########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

    print( 'sucessful!')