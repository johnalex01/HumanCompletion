from Processor import Processor
from BiProcessor import BiProcessor
from BiProcessor_noRes import BiProcessor_noRes
from BiProcessor_for_single_side_attention import BiProcessor_for_single_side_attention
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # train_dataset = MyDataset('../datasets/new_human36_3D/npy_data/total_train_ab.npy', mode_name='train')
    # train_loader = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=32)
    #
    # for data in train_loader:
    #     print(data['pre_10'].shape)
    #     d = -1

    ss = Processor()
    #ss = BiProcessor()
    #ss = BiProcessor_noRes()
    
    #ss = BiProcessor_for_single_side_attention()
    ss.train_and_test()


    # test_dataset = DatasetTest('../datasets/new_human36_3D/test/test_npy/', 'directions', global_max=1000, global_min=-1000)
    # d = -1

    pass
