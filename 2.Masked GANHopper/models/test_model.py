import torch
from .base_model import BaseModel
from . import networks
from thop import profile
import copy

class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(no_dropout=True)
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        parser.add_argument('--n_labels', type=int, default=7, help='number of labels in the mask')
        parser.add_argument('--num_hops', type=int, default=4, help='number of hops')
        
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_' + opt.model_suffix]#, 'mask_0', 'mask_1', 'mask_2']
        for i in range(self.opt.num_hops):
            self.visual_names.append('hop_' + str(i+1) + '_' + opt.model_suffix)
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G_' + opt.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(opt.n_labels * opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG_' + opt.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input, lambda_phase_out=[]):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real = input['A'].to(self.device)
        
        nc = self.opt.n_labels
        self.labe = input['Amask'].to(self.device)
        self.labels = self.labe.mul(255).long()
        bs, _ , h, w = self.labels.size()
        label = torch.zeros(bs, nc, h, w).to(self.device)
        self.mask = label.scatter_(1, self.labels, 1.0)
        
        self.mask_divided = []
        for i in range(self.opt.n_labels):
            self.mask_divided.append(self.mask.detach().narrow(1, i, 1))
        self.mask_0 = self.mask.detach().narrow(1, 0, 1)
        self.mask_1 = self.mask.detach().narrow(1, 1, 3)
        self.mask_2 = self.mask.detach().narrow(1, 4, 3)
        
        self.real_concat = self.real * self.mask_divided[0]
        for i in range(1, self.opt.n_labels):
            self.real_concat = torch.cat((self.real_concat, self.real * self.mask_divided[i]), 1)

        setattr(self, 'real_' + self.opt.model_suffix, self.real)
        self.prev = self.real.detach()
        self.prev_concat = self.real_concat.detach()

        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass."""
        a1 = 0.0
        a2 = 0.0
        # num, _ = count_ops(self.netG_A, self.prev_A_concat)
        
        # x = torch.tensor(0.0)
        # x = x.to(self.device)
        # print ("cuda")
        # print (self.prev_concat.is_cuda)
        # print (x.is_cuda)
        # # print (self.netG.is_cuda)
        # with torch.no_grad():
        #     macs, params = profile(copy.deepcopy(self.netG), inputs=(self.prev_concat, x, x))
        # print ("FLOPS:", macs, "PARAMS:", params)

        
        for hop_idx in range(self.opt.num_hops):
            self.next = self.netG(self.prev_concat, a1, a2)  # G(A)
            
            self.next_concat = self.next * self.mask_divided[0]
            for i in range(1, self.opt.n_labels):
                self.next_concat = torch.cat((self.next_concat, self.next * self.mask_divided[i]), 1)
            
            setattr(self, 'hop_' + str(hop_idx+1) + '_' + self.opt.model_suffix, self.next)
            self.prev = self.next.detach()
            self.prev_concat = self.next_concat.detach()
        

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
