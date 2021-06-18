import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from thop import profile
import copy


class MaskedGANHopperModel(BaseModel):
    """
    This class implements the Masked GANHopper model (based on the Cycle GANmodel), for learning masked image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_12blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_hybrid', type=float, default=1.0, help='weight for hybrid loss')
            parser.add_argument('--thresh_hybrid', type=float, default=0.002, help='threshold to stop training hybrid loss')
            parser.add_argument('--lambda_smooth', type=float, default=1.0, help='weight for smoothness loss')
            parser.add_argument('--num_prev_losses', type=int, default=100, help='number of losses for G_A and G_B to take average from for progressive')
        
        parser.add_argument('--n_labels', type=int, default=7, help='number of labels in the mask')
        parser.add_argument('--num_hops', type=int, default=4, help='number of hops')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.base_loss_names = ['D_A', 'G_A', 'cycle_A', 'hybrid_A', 'smooth_A', 'D_B', 'G_B', 'cycle_B', 'hybrid_B', 'smooth_B', 'D_H']
        self.loss_names = []
        for i in range(opt.num_hops):
            for name in self.base_loss_names:
                self.loss_names.append(name + '_' + str(i+1))
        print (self.loss_names)
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A']
        visual_names_B = ['real_B']#, 'real_B_l0', 'real_B_l1', 'real_B_l2', 'real_B_l3', 'real_B_l4', 'real_B_l5', 'real_B_l6']
        for i in range(self.opt.num_hops):
            visual_names_A.append('hop_A_' + str(i+1))
            visual_names_B.append('hop_B_' + str(i+1))
            visual_names_A.append('rec_A_' + str(i+1))
            visual_names_B.append('rec_B_' + str(i+1))
    

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'D_H']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.n_labels * opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.n_labels * opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.small = networks.small()
        self.big = networks.big()

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_H = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                        opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
        if self.isTrain:
            
            # create image buffer to store previously generated images
            self.next_A_pool = ImagePool(opt.pool_size)
            self.next_B_pool = ImagePool(opt.pool_size)
            
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_H.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.flag1 = False
            self.flag2 = False
            self.loss_ind = 0
            self.prev_losses = [100 for _ in range(opt.num_prev_losses)]

    def reset_prev_losses(self):
        self.prev_losses = [100 for _ in range(self.opt.num_prev_losses)]

    def set_input(self, input, lambda_phase_out):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        a1 = lambda_phase_out[0]
        a2 = lambda_phase_out[1]
        AtoB = self.opt.direction == 'AtoB'
        nc = self.opt.n_labels
        self.real_A_0 = input['A' if AtoB else 'B'].to(self.device)
        self.real_B_0 = input['B' if AtoB else 'A'].to(self.device)
        self.labe_A = input['Amask' if AtoB else 'Bmask'].to(self.device)
        self.labe_B = input['Bmask' if AtoB else 'Amask'].to(self.device)

        self.labels_A = self.labe_A.mul(255).long()
        self.labels_B = self.labe_B.mul(255).long()
        
        bs, _ , h, w = self.labels_A.size()
        label_A = torch.zeros(bs, nc, h, w).to(self.device)
        label_B = torch.zeros(bs, nc, h, w).to(self.device)
        self.mask_A = label_A.scatter_(1, self.labels_A, 1.0)
        self.mask_B = label_B.scatter_(1, self.labels_B, 1.0)

        self.real_A_1 = self.big(self.small(self.real_A_0))
        self.real_A_2 = self.big(self.big(self.small(self.small(self.real_A_0))))
        self.real_A = a1 * self.real_A_2 + (1-a1) * (a2 * self.real_A_1 + (1-a2) * self.real_A_0)

        self.real_B_1 = self.big(self.small(self.real_B_0))
        self.real_B_2 = self.big(self.big(self.small(self.small(self.real_B_0))))
        self.real_B = a1 * self.real_B_2 + (1-a1) * (a2 * self.real_B_1 + (1-a2) * self.real_B_0)

        # self.mask_A_1 = self.big(self.small(self.mask_A_0))
        # self.mask_A_2 = self.big(self.big(self.small(self.small(self.mask_A_0))))
        # self.mask_A = a1 * self.mask_A_2 + (1-a1) * (a2 * self.mask_A_1 + (1-a2) * self.mask_A_0)
        
        # self.mask_B_1 = self.big(self.small(self.mask_B_0))
        # self.mask_B_2 = self.big(self.big(self.small(self.small(self.mask_B_0))))
        # self.mask_B = a1 * self.mask_B_2 + (1-a1) * (a2 * self.mask_B_1 + (1-a2) * self.mask_B_0)

        self.mask_A_divided = []
        self.mask_B_divided = []
        for i in range(self.opt.n_labels):
            self.mask_A_divided.append(self.mask_A.detach().narrow(1, i, 1))
            self.mask_B_divided.append(self.mask_B.detach().narrow(1, i, 1))

        self.real_A_concat = self.real_A * self.mask_A_divided[0]
        self.real_B_concat = self.real_B * self.mask_B_divided[0]
        for i in range(1, self.opt.n_labels):
            self.real_A_concat = torch.cat((self.real_A_concat, self.real_A * self.mask_A_divided[i]), 1)
            self.real_B_concat = torch.cat((self.real_B_concat, self.real_B * self.mask_B_divided[i]), 1)
        
        self.prev_A = self.real_A.detach()
        self.prev_B = self.real_B.detach()
        self.prev_A_concat = self.real_A_concat.detach()
        self.prev_B_concat = self.real_B_concat.detach()

        # self.lambda_label = []
        # for i in range(self.opt.n_labels):
        #     self.lambda_label.append(1.0)
        # self.lambda_label[1] = 3.0
        # self.lambda_label[2] = 3.0
        self.lambda_label = self.opt.rec_weights
        # print (self.opt.rec_weights)
        # print(self.lambda_label)

        # self.real_B_l0 = self.real_B * self.mask_B_divided[0]
        # self.real_B_l1 = self.real_B * self.mask_B_divided[1]
        # self.real_B_l2 = self.real_B * self.mask_B_divided[2]
        # self.real_B_l3 = self.real_B * self.mask_B_divided[3]
        # self.real_B_l4 = self.real_B * self.mask_B_divided[4]
        # self.real_B_l5 = self.real_B * self.mask_B_divided[5]
        # self.real_B_l6 = self.real_B * self.mask_B_divided[6]

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        
    def forward(self, a1, a2):
    
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.next_A = self.netG_A(self.prev_A_concat, a1, a2)  # G_A(A)
        self.next_A_concat = self.next_A * self.mask_A_divided[0]
        for i in range(1, self.opt.n_labels):
            self.next_A_concat = torch.cat((self.next_A_concat, self.next_A * self.mask_A_divided[i]), 1)
        self.rec_prev_A = self.netG_B(self.next_A_concat, a1, a2)
        
        self.next_B = self.netG_B(self.prev_B_concat, a1, a2)  # G_A(A)
        self.next_B_concat = self.next_B * self.mask_B_divided[0]
        for i in range(1, self.opt.n_labels):
            self.next_B_concat = torch.cat((self.next_B_concat, self.next_B * self.mask_B_divided[i]), 1)
        self.rec_prev_B = self.netG_A(self.next_B_concat, a1, a2)
            

    def backward_D_basic(self, netD, real, fake, a1, a2):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real, a1, a2)
        loss_D_real = self.criterionGAN(pred_real, 1.0)
        # Fake
        pred_fake = netD(fake.detach(), a1, a2)
        loss_D_fake = self.criterionGAN(pred_fake, 0.0)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self, a1, a2):
        """Calculate GAN loss for discriminator D_A"""
        next_A = self.next_A_pool.query(self.next_A)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, next_A, a1, a2)

    def backward_D_B(self, a1, a2):
        """Calculate GAN loss for discriminator D_B"""
        next_B = self.next_B_pool.query(self.next_B)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, next_B, a1, a2)
    
    def backward_D_H(self, a1, a2):
        """Calculate GAN loss for discriminator D_H"""
        self.loss_D_H = self.backward_D_basic(self.netD_H, self.real_B, self.real_A, a1, a2)
        if self.loss_D_H < self.opt.thresh_hybrid:
            if not self.flag1:
                self.flag1 = True
            elif not self.flag2: 
                self.flag2 = True

    def proportional_loss(self, x, y, mask):
        total = 0
        loss = 0.0
        # loss_tmp = 0
        for i in range(self.opt.n_labels):
            num = torch.sum(mask[i])
            total += num
            if num > 0.1:
                dummy = self.criterionCycle(x * mask[i], y * mask[i]) 
                #print ("it was:", dummy)
                dummy *= self.lambda_label[i]
                #print ("it is:", dummy)
                loss = loss +  dummy / num
                # loss_tmp += dummy
        # tmploss = self.criterionCycle(x, y)
        # print ("total is:", total, "loss is:", loss*total, "normal loss was:", tmploss, "sum is:", loss_tmp)
        # self.tmp1 = x * mask[0]
        # self.tmp2 = y * mask[0]
        total /= self.opt.n_labels
        return loss*total

    def backward_G(self, hop_idx, a1, a2):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_hybrid = self.opt.lambda_hybrid
        lambda_smooth = self.opt.lambda_smooth
        num_hops = self.opt.num_hops
        
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.next_A, a1, a2), 1.0)  
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.next_B, a1, a2), 1.0)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.proportional_loss(self.rec_prev_A, self.prev_A, self.mask_A_divided) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||    
        self.loss_cycle_B = self.proportional_loss(self.rec_prev_B, self.prev_B, self.mask_B_divided) * lambda_B
        if lambda_hybrid == 0:
            self.loss_hybrid_A = 0
            self.loss_hybrid_B = 0
        else:
            # Forward hybrid domain loss
            self.loss_hybrid_A = self.criterionGAN(self.netD_H(self.next_A, a1, a2), hop_idx/num_hops) * lambda_hybrid
            # Backward hybrid domain loss
            self.loss_hybrid_B = self.criterionGAN(self.netD_H(self.next_B, a1, a2), (num_hops - hop_idx)/num_hops) * lambda_hybrid
        if lambda_smooth == 0: 
            self.loss_smooth_A = 0
            self.loss_smooth_B = 0
        else:
            # Forward smooth loss
            self.loss_smooth_A = self.proportional_loss(self.next_A, self.prev_A, self.mask_A_divided) * lambda_smooth
            # Backward smooth loss
            self.loss_smooth_B = self.proportional_loss(self.next_B, self.prev_B, self.mask_B_divided) * lambda_smooth

        # combined loss and calculate gradients
        self.loss_G =  \
            self.loss_G_A + self.loss_G_B + \
            self.loss_cycle_A + self.loss_cycle_B + \
            self.loss_hybrid_A + self.loss_hybrid_B + \
            self.loss_smooth_A + self.loss_smooth_B
        if hop_idx == self.opt.num_hops:
            self.prev_losses[self.loss_ind] = self.loss_G_A + self.loss_G_B 
            self.loss_ind = (self.loss_ind + 1) % self.opt.num_prev_losses
        self.loss_G.backward()

    def set_visuals(self, idx):
        name_A = 'hop_A_' + str(idx)
        name_B = 'hop_B_' + str(idx)
        name_rec_A = 'rec_A_' + str(idx)
        name_rec_B = 'rec_B_' + str(idx)
        setattr(self, name_A, self.next_A)
        setattr(self, name_B, self.next_B)
        setattr(self, name_rec_A, self.rec_prev_A)
        setattr(self, name_rec_B, self.rec_prev_B)

    def set_losses(self, idx):
        for name in self.base_loss_names:
            setattr(self, 'loss_' + name + '_' + str(idx), getattr(self, 'loss_' + name))
        
    def optimize_parameters(self, lambda_phase_out):
        a1 = lambda_phase_out[0]
        a2 = lambda_phase_out[1]
        # x = torch.tensor(0.0)
        # x = x.to(self.device)
        # print ("cuda")
        # print (self.prev_A_concat.is_cuda)
        # print (x.is_cuda)
        # print (self.prev_A_concat.shape)
        # # print (self.netG.is_cuda)
        # with torch.no_grad():
        #     macs, params = profile(copy.deepcopy(self.netG_A), inputs=(self.prev_A_concat, x, x))
        # print ("FLOPS:", macs, "PARAMS:", params)
        
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        num_hops = self.opt.num_hops
        for hop_idx in range(num_hops):
            # forward
            self.forward(a1, a2)                                                     # compute fake images and reconstruction images.
            
            # G_A and G_B
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_H], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_G.zero_grad()                                            # set G_A and G_B's gradients to zero
            self.backward_G(hop_idx+1, a1, a2)                                              # calculate gradients for G_A and G_B
            self.optimizer_G.step()                                                 # update G_A and G_B's weights
            
            # D_A and D_B
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_H], True)
            self.optimizer_D.zero_grad()                                            # set D_A and D_B's gradients to zero
            if self.flag2:
                self.set_requires_grad([self.netD_H], False)
            else:
                self.backward_D_H(a1, a2)                                                 # calculate graidents for D_H
            self.backward_D_A(a1, a2)                                                     # calculate gradients for D_A
            self.backward_D_B(a1, a2)                                                     # calculate graidents for D_B
            self.optimizer_D.step()                                                 # update D_A and D_B's weights
            
            self.prev_A = self.next_A.detach()
            self.prev_B = self.next_B.detach()
            self.prev_A_concat = self.next_A_concat.detach()
            self.prev_B_concat = self.next_B_concat.detach()

            self.set_visuals(hop_idx+1)
            self.set_losses(hop_idx+1)

    def test_hopper(self, a1, a2):
        num_hops = self.opt.num_hops
        with torch.no_grad():
            for hop_idx in range(num_hops):
                # forward
                self.forward(a1, a2)      # compute fake images and reconstruction images.
                self.prev_A = self.next_A.detach()
                self.prev_B = self.next_B.detach()
                self.prev_A_concat = self.next_A_concat.detach()
                self.prev_B_concat = self.next_B_concat.detach()

                self.set_visuals(hop_idx+1)
                

