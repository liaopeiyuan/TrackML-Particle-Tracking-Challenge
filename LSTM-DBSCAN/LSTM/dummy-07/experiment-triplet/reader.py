from common import *

from dataset.transform import *
from dataset.sampler import *
from dataset.others import *

from utility.file import *
from utility.draw import *



#data reader  ----------------------------------------------------------------

class DummyDataset(Dataset):

    def __init__(self, split, transform=None, mode='train'):
        super(DummyDataset, self).__init__()
        self.transform=transform
        self.mode=mode
        self.split=split
        self.samples = load_pickle_file(
            '/root/share/project/kaggle/cern/data/%s.pickle'%split
        )


    def __getitem__(self, index):
        event, data, combination, label, = self.samples[index]

        if self.transform is not None:
            return self.transform( data, combination, label, index)
        else:
            return  data, combination, label, index


    def __len__(self):
        return len(self.samples)



# check ##################################################################################3

def run_check_train_dataset_reader():


    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(111, projection='3d')

    #dataset = CernDataset(
    dataset = DummyDataset(
        'samples_train',
        mode='<not_used>',transform = None,
    )
    print(len(dataset))

    for n in range(len(dataset)):
        data, combination, label, index = dataset[n]

        ax.clear()
        x = data.x.values
        y = data.y.values
        z = data.z.values
        ax.plot(x,y,z,'.',color = [0.75,0.75,0.75], markersize=3)

        #sample positive
        pos = np.where(label==1)[0]
        for i in pos:
            t = combination[i]
            color=np.random.uniform(0,1,(3))
            ax.plot(t[:,0],t[:,1],t[:,2],'.-',markersize=8,color=color)

        set_figure(ax,x_limit=(0,-100),y_limit=(-20,20),z_limit=(500,1000))
        plt.pause(5)
        #plt.show()


    plt.show()







# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_train_dataset_reader()

    print( 'sucess!')













