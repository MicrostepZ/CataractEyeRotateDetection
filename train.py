# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os, pdb
import torch
import torch.optim as optim
from torch.utils import tensorboard
from tools import common, trainer
from tools.dataloader import *
from nets.patchnet import *
from nets.ournet import *
from nets.losses import *

default_net = "ournet()"
#default_net = "Fast_Quad_L2Net_ConfCFS()"

db_cataract = """SyntheticPairDataset(
            ImgFolder('data/Cataract_Train/Train/'), 
            'RandomScale(256,1024,can_upscale=True)', 
            'RandomTilting(0.5), PixelNoise(25)')"""

db_web_images = """SyntheticPairDataset(
        web_images, 
        'RandomScale(256,1024,can_upscale=True)',
        'RandomTilting(0.5), PixelNoise(25)')"""

db_aachen_images = """SyntheticPairDataset(
        aachen_db_images, 
        'RandomScale(256,1024,can_upscale=True)', 
        'RandomTilting(0.5), PixelNoise(25)')"""

db_aachen_style_transfer = """TransformedPairs(
            aachen_style_transfer_pairs,
            'RandomScale(256,1024,can_upscale=True), RandomTilting(0.5), PixelNoise(25)')"""

db_aachen_flow = "aachen_flow_pairs"

data_sources = dict(
    D = db_cataract,
    W = db_web_images,
    A = db_aachen_images,
    F = db_aachen_flow,
    S = db_aachen_style_transfer,
    )

default_dataloader = """PairLoader(CatPairDataset(`data`),
    scale   = 'RandomScale(256,1024,can_upscale=True)',
    distort = 'ColorJitter(0.2,0.2,0.2,0.1)',
    crop    = 'RandomCrop(192)')"""

default_sampler = """NghSampler2(ngh=7, subq=-8, subd=1, pos_d=3, neg_d=5, border=16,
                            subd_neg=-8,maxpool_pos=True)"""

default_loss = """MultiLoss(
        1, ReliabilityLoss(`sampler`, base=0.5, nq=20),
        1, CosimLoss(N=`N`),
        1, PeakyLoss(N=`N`))"""


class MyTrainer(trainer.Trainer):
    """ This class implements the network training.
        Below is the function I need to overload to explain how to do the backprop.
    """
    def forward_backward(self, inputs):
        output = self.net(imgs=[inputs.pop('img1'),inputs.pop('img2')])
        allvars = dict(inputs, **output)
        loss, details = self.loss_func(**allvars)
        if torch.is_grad_enabled(): loss.backward()
        return loss, details



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Train Net")

    parser.add_argument("--data-loader", type=str, default=default_dataloader)
    parser.add_argument("--train-data", type=str, default=list('DWAF'), nargs='+',choices = set(data_sources.keys()),help="Pretrain_Set: WAF; Finetune_Set: DWAF")
    parser.add_argument("--net", type=str, default=default_net, help='network architecture')

    parser.add_argument("--pretrained", type=str, default="", help='pretrained model path')
    parser.add_argument("--save_path", type=str, default='logs/train_ours/', help='model save_path path')
    
    parser.add_argument("--loss", type=str, default=default_loss, help="loss function")
    parser.add_argument("--sampler", type=str, default=default_sampler, help="AP sampler")
    parser.add_argument("--N", type=int, default=16, help="patch size for repeatability")

    parser.add_argument("--epochs", type=int, default=20, help='number of training epochs')
    parser.add_argument("--batch-size", "--bs", type=int, default=6, help="batch size")
    parser.add_argument("--learning-rate", "--lr", type=str, default=1e-4)
    parser.add_argument("--weight-decay", "--wd", type=float, default=5e-4)
    
    parser.add_argument("--threads", type=int, default=8, help='number of worker threads')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='-1 for CPU')
    
    args = parser.parse_args()
    
    iscuda = common.torch_set_gpu(args.gpu)
    common.mkdir_for(args.save_path)

    # Tensorboard
    writer_dir = os.path.join(args.save_path, 'runs')
    writer = tensorboard.SummaryWriter(writer_dir)
    os.makedirs(os.path.join(args.save_path, 'checkpoints'), exist_ok=True)
    file = open(os.path.join(args.save_path, 'args.txt'), 'w')
    file.write(str(args))
    file.close()

    best_loss = 10

    # Create data loader
    from datasets import *
    db = [data_sources[key] for key in args.train_data]
    db = eval(args.data_loader.replace('`data`', ','.join(db)).replace('\n', ''))
    print("Training image database =", db)
    loader = threaded_loader(db, iscuda, args.threads, args.batch_size, shuffle=True)

    # create network
    print("\n>> Creating net = " + args.net) 
    net = eval(args.net)
    print(f" ( Model size: {common.model_size(net)/1000:.0f}K parameters )")

    # initialization
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, lambda a,b:a)
        net.load_state_dict(checkpoint['state_dict'])
        
    # create losses
    loss = args.loss.replace('`sampler`',args.sampler).replace('`N`',str(args.N))
    print("\n>> Creating loss = " + loss)
    loss = eval(loss.replace('\n',''))
    
    # create optimizer
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad],
                            lr=args.learning_rate, weight_decay=args.weight_decay)

    train = MyTrainer(net, loader, loss, optimizer)
    if iscuda: train = train.cuda()

    # Training loop #
    for epoch in range(args.epochs):
        print(f"\n>> Starting epoch {epoch}...")
        loss_reliability, loss_cosim16, loss_peaky16, totalloss = train()
        writer.add_scalar('Train/train_loss_reliability', loss_reliability, epoch)
        writer.add_scalar('Train/train_loss_cosim16', loss_cosim16, epoch)
        writer.add_scalar('Train/train_loss_peaky16', loss_peaky16, epoch)
        writer.add_scalar('Train/train_totalloss', totalloss, epoch)
        sava_path = os.path.join(args.save_path, 'checkpoints', str(epoch) + '_epoch_model.pt')
        print(f"\n>> Saving model to {sava_path}")
        torch.save({'net': args.net, 'state_dict': net.state_dict()}, sava_path)
        if totalloss < best_loss:
            best_loss = totalloss
            sava_best_path = os.path.join(args.save_path, 'checkpoints', 'best_model.pt')
            torch.save({'net': args.net, 'state_dict': net.state_dict()}, sava_best_path)
            print(f"\n>> Saving best model to {sava_best_path}")


