from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_mask
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        #self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        transform_paramsA = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_paramsA)
        A = A_transform(A_img)

        delimiterA_pos = A_path.rfind("/")
        Amask_path = A_path[:delimiterA_pos]+"_mask"+A_path[delimiterA_pos:-3]+"png"
        A_mask = Image.open(Amask_path)
        Amask_transform = get_transform_mask(self.opt, transform_paramsA)
        Amask = Amask_transform(A_mask)
        return {'A': A, 'Amask': Amask, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
