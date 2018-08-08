from common import *




## block ## -------------------------------------

class Linear_Bn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear_Bn, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels,bias=False)
        self.bn   = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return x


## net  ## -------------------------------------
class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        num_points = 3


        self.feature = nn.Sequential(
            Linear_Bn(3*num_points, 64),
            nn.PReLU(num_parameters=1, init=0.25),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.1, inplace=False),
            Linear_Bn( 64,  128),
            nn.PReLU(num_parameters=1, init=0.25),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.1, inplace=False),
            Linear_Bn(128,  256),
            nn.PReLU(num_parameters=1, init=0.25),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1, inplace=False),
            Linear_Bn(256,  512),
            nn.PReLU(num_parameters=1, init=0.25),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.1, inplace=False),
            Linear_Bn(512,  1024),
            nn.PReLU(num_parameters=1, init=0.25),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.1, inplace=False),

            Linear_Bn(1024, 2048),
            nn.PReLU(num_parameters=1, init=0.25),
            nn.BatchNorm1d(2048),
            Linear_Bn(2048, 1024),
            nn.PReLU(num_parameters=1, init=0.25),
            nn.BatchNorm1d(1024),
            Linear_Bn(1024, 512),
            nn.PReLU(num_parameters=1, init=0.25),
            nn.BatchNorm1d(512),
            Linear_Bn(512,  256),
            nn.PReLU(num_parameters=1, init=0.25),
            nn.BatchNorm1d(256),
            Linear_Bn(256,  128),
            nn.PReLU(num_parameters=1, init=0.25),
            nn.BatchNorm1d(128),
            Linear_Bn(128,   64),
            nn.PReLU(num_parameters=1, init=0.25),
        )

        self.logit = nn.Sequential(
            nn.Linear(64, 1)
        )
        # self.target = nn.Sequential(
        #     nn.Linear(64, 3)
        # )


    def forward(self, x):

        batch_size  = x.size(0)
        x = x.view(batch_size,-1)

        x      = self.feature(x)
        logit  = self.logit(x).view(-1)

        return logit


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

    #create dummy data
    batch_size  = 5
    num_points  = 100
    tracklet = torch.randn((batch_size,num_points,3))
    tracklet = tracklet.cuda()

    net = TripletNet().cuda()
    logit = net(tracklet)

    # print(type(net))
    # print(net,'\n')

    print(logit,'\n')
    print(logit.size(),'\n')


    print('')





########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

    print( 'sucessful!')