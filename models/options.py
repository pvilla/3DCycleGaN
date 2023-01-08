import argparse
from datetime import datetime
import os
class ParamOptions():
    """This class defines options used during both training and test time."""
    def __init__(self):
        self.initialized = False
        self.time = datetime.now()
        self.cwd = os.getcwd()

    def initialize(self,parser):
        parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='unpaired phase reconstruction using propagation-enhanced cycle-consistent adversarial network')
        parser.add_argument('--run_path', type=str, default = F'{self.cwd}/results', help='path to save results')
        parser.add_argument('--run_name', type=str, default = self.time.strftime("%y_%m_%d-%H_%M"), help='folder name of this run') #TODO: modify save_path and run_name
        parser.add_argument('--batch_size', '-b', type=int, default=8, help='input batch size')
        parser.add_argument('--data_A', type=str, default='dataset/T700_GF_t05_pxs08.json', help='json file with datasets')
        parser.add_argument('--data_B', type=str, default='dataset/T700_pco_pxs04.json', help='json file with datasets')
        parser.add_argument('--lambda_GA', type=float, default=1.0, help='weight for adversarial loss of generator A')
        parser.add_argument('--lambda_GB', type=float, default=1.0, help='weight for adversarial loss of generator B')
        parser.add_argument('--lambda_FSCA', type=float, default=0.0, help='weight for Fourier ring correlation loss A')
        parser.add_argument('--lambda_FSCB', type=float, default=0.0, help='weight for Fourier ring correlation loss B')
        parser.add_argument('--lambda_A', nargs='+', type=float, default=10.0, help='weight for cycle consistency loss between real_A and rec_A')
        parser.add_argument('--lambda_B', nargs='+', type=float, default=10.0, help='weight for cycle consistency loss between real_B and rec_B')
        parser.add_argument('--pretrainpath', type=str, default='None', help='pretrained networks. define a path with pretrained networks. default it is pretrained with vgg11')
        parser.add_argument('--pretrained', type=str, default='None', help='pretrained networks. define a path with pretrained networks. default it is pretrained with vgg11')
        parser.add_argument('--loadnetpath', type=str, default='False', help='pretrained networks. define a path with pretrained networks. default it is pretrained with vgg11')
        parser.add_argument('--loadnetepoch', type=int, default=0, help='pretrained networks. define a path with pretrained networks. default it is pretrained with vgg11')
        parser.add_argument('--channels_A', type=int, default=1, help='number of input channels domain A')
        parser.add_argument('--channels_B', type=int, default=1, help='number of input channels domain B')
        parser.add_argument('--isTest', action='store_true', help='not train the model')
        parser.add_argument('--propagator_A', type=str, default='None', help='manipulation in the A-cycle')
        parser.add_argument('--propagator_B', type=str, default='None', help='manipulation in the B-cycle')
        parser.add_argument('--lr_g', type=float, default=0.0002, help='initial learning rate for the generator')
        parser.add_argument('--lr_d', type=float, default=0.0001, help='initial learning rate for the discriminator')
        parser.add_argument('--num_epochs','-n', type=int, default=1000, help='total number of epochs')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--clip_max', type=float, default=1.0, help='maximum value for the gradient clipping, set to 0 if do not want to use gradient clipping.')
        parser.add_argument('--image_stats', type=list,default=[0,1,0,1,0,1], help='statistics of training images written as [real_A_mean, real_A_std, real_B_ch1_mean, real_B_ch1_std, real_B_ch2_mean, real_B_ch2_std]')
        parser.add_argument('--imsize_A', nargs='+', type=int, default=[128,128,128], help='window size in the A domain') 
        parser.add_argument('--crop_mode', type=str, default='center', help='Center, Sliding or Random')      
        parser.add_argument('--adjust_lr_epoch', type=int, default=20, help='set the learning rate to the initial learning rate decayed by 10 every certain epochs')
        parser.add_argument('--log_note', type=str, default=' ', help='run note which will be saved to the log file')
        parser.add_argument('--save_model_freq_epoch', type=int, default=1, help='frequency of saving models (epoch)')
        parser.add_argument('--save_model_start_epoch', type=int, default=1, help='frequency of saving models (epoch)')
        parser.add_argument('--print_loss_freq_iter', type=int, default=5, help='frequency of print loss (iteration)')
        parser.add_argument('--save_cycleplot_freq_iter', type=int, default=20, help='freqency of save cycle plots for train images')
        parser.add_argument('--save_val_freq_epoch',type=int, default=1, help='frequency of save cycle plots for validation images')
        parser.add_argument('--net_A', type=str,default='UNet11', help='function name of the generator net in the A cycle')
        parser.add_argument('--net_B', type=str,default='UNet11', help='function name of the generator net in the B cycle')
        parser.add_argument('--super_resolution', type=float, default=1.0, help='magnification from domain A to domain B')
        # visdom parameters
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized: 
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,add_help=False)
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        opt = self.gather_options()
        
        self.opt = opt
        return self.opt
