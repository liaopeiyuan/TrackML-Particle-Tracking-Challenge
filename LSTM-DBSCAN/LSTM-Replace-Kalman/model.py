from common import *


# https://indico.cern.ch/event/658267/contributions/2881175/attachments/1621912/2581064/Farrell_heptrkx_ctd2018.pdf
class LstmNet(nn.Module):
    def __init__(self, in_channels=3, num_estimate=2 ):
        super(LstmNet, self).__init__()
        self.lstm   = nn.LSTM(in_channels, 24, batch_first=True)
        self.linear = nn.Linear(24, num_estimate)


    def forward(self, point):
        batch_size, num_point, dim = point.size()

        x,(hidden, cell) = self.lstm(point)
        x = self.linear(x)

        return x


    def criterion(self, estimate, truth):
        estimate = estimate.view(-1)
        truth = truth.view(-1)
        loss  = F.mse_loss(estimate, truth, size_average=True)
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

    #create dummy data
    batch_size   = 10
    num_point    = 20
    dim          = 3
    point        = np.random.uniform(-1,1,(batch_size,num_point,dim)).astype(np.float32)

    ##----------------------------------------------
    point = torch.from_numpy(point).cuda()



    net = LstmNet().cuda()
    estimate = net(point)
    #print(estimate)
    print(estimate.size())

    truth = np.random.uniform(-1,1,(batch_size,num_point,2)).astype(np.float32)
    truth = torch.from_numpy(truth).cuda()

    loss = net.criterion(estimate,truth)
    print(loss)






########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()


    print( 'sucessful!')