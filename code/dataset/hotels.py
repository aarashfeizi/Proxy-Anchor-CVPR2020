from .base import *
from tqdm import tqdm

class Hotels(BaseDataset):
    def __init__(self, root, mode, transform=None):
        self.mode = mode
        if mode == 'train':
            self.root = root + '/hotels50k_v5_restructured/train_small'
        elif self.mode == 'eval':
            self.root = root + '/hotels50k_v5_restructured/val1_small'
        self.transform = transform
        print('getting classes')
        self.classes = torchvision.datasets.ImageFolder(root=root).classes
        # if self.mode == 'train':
        #     self.classes = range(0, 100)
        # elif self.mode == 'eval':
        #     self.classes = range(100, 200)

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        index = 0
        print('getting imgs...')
        with tqdm(total=len(torchvision.datasets.ImageFolder(root=self.root).imgs), desc=f'Loading hotels {mode}...') as t:
            print('getting imgs 2...')
            for i in torchvision.datasets.ImageFolder(root=self.root).imgs:
                # i[1]: label, i[0]: root
                y = i[0].split('/')[-2]
                # fn needed for removing non-images starting with `._`
                fn = os.path.split(i[0])[1]
                if y in self.classes and fn[:2] != '._':
                    self.ys += [y]
                    self.I += [index]
                    self.im_paths.append(os.path.join(self.root, i[0]))
                    index += 1

            t.update()