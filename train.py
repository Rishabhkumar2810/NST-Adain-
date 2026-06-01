import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from utils.utils import *
from utils.models import * 
import torch.optim as optim 
from tqdm import tqdm
from torchvision.utils import save_image

def parse_arguements():
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--content_dir',type=str,default=r'C:\Users\risha\Desktop\NST\content_data',help='Location of content data')
    
    parser.add_argument('--style_dir',type=str,default=r'C:\Users\risha\Desktop\NST\style_data',help='Location of style data')
   
    parser.add_argument('--vgg',type=str,default=r'C:\Users\risha\Desktop\NST\vgg_normalised.pth',help='Location of pre_trained VGG')
  
    parser.add_argument('--experiment',type=str,default='experiment1',help='Name of Experiment')

    parser.add_argument('--final_size',type=int,default=256,help='Size of final image')

    parser.add_argument('--content_size',type=int,default=512,help='Size of content image')

    parser.add_argument('--style_size',type=int,default=512,help='Size of style image')

    parser.add_argument('--crop',action='store_true',default= True,help='Crop the image')
  
    parser.add_argument('--batch_size',type=int,default=4,help='Batch size for training')

    parser.add_argument('--lr',type=float,default=1e-4,help='Learning rate')
    
    parser.add_argument('--lr_decay',type=float,default=5e-5,help='Learning rate decay')

    parser.add_argument('--epochs',type=int,default=1,help='Number of epochs for training') 

    parser.add_argument('--content_weight',type=float,default=1.0,help='Weight for content loss')
    
    parser.add_argument('--style_weight',type=float,default=10,help='Weight for style loss')

    parser.add_argument('--log_interval',type=int,default=1,help='Interval for logging training status')

    parser.add_argument('--save_interval',type=int,default=2,help='Interval for saving model checkpoints')
    
    parser.add_argument('--resume',action='store_true',default=False,help='Resume training from checkpoint')

    parser.add_argument('--decoder_path',type=str,default=None,help='Path to decoder checkpoint for resuming training')

    parser.add_argument('--optimizer_path',type=str,default=None,help='Path to optimizer checkpoint for resuming training')
    return parser.parse_args()



def main():
    args = parse_arguements()
    print(args)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir=Path('experiment')/args.experiment
    save_dir.mkdir(parents=True,exist_ok=True)

    #Save  arguments value
    with open(save_dir/'args.txt','w') as args_file:
        for key,value in vars(args).items():
            args_file.write(f'{key}:{value}\n')
  
  
  
    content_transform =get_transform(args.content_size,args.crop,args.final_size)
    style_transform =get_transform(args.style_size,args.crop,args.final_size)
   
   
    content_dataset=ImageFolderDataset(args.content_dir,transform=content_transform)
    style_dataset=ImageFolderDataset(args.style_dir,transform=style_transform)

    content_loader=DataLoader(content_dataset,batch_size=args.batch_size,shuffle=False,pin_memory=False,drop_last=True)
    style_loader=DataLoader(style_dataset,batch_size=args.batch_size,shuffle=False,pin_memory=False,drop_last=True)

    encoder =VGGEncoder(args.vgg).to(device)
    decoder =Decoder().to(device)


    optimizer=optim.Adam(decoder.parameters(),lr=args.lr )

    scheduler=optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch: 1.0 /(1.0+args.lr_decay*epoch))

    if args.resume:
        decoder. load_state_dict(torch.load(args.decoder_path))
        optimizer.load_state_dict(torch.load(args.optimizer_path))
    mse_loss=torch.nn.MSELoss()
    encoder.eval()

    running_loss=None
    running_closs=None
    running_sloss=None
    for epoch in range(args.epochs):
        
        progress_bar=tqdm(zip(content_loader,style_loader),
                          total=min(len(content_loader),len(style_loader)))
        
        running_loss=0
        running_closs=0
        running_sloss=0


        for content_batch,style_batch in progress_bar:
            content_batch=content_batch.to(device)
            style_batch=style_batch.to(device)

            c_feats=encoder(content_batch)
            s_feats=encoder(style_batch)

            t=adaptive_instance_normalization(c_feats[-1],s_feats[-1])
            
            g=decoder(t)

            g_feats=encoder(g)

            loss_c=mse_loss(g_feats[-1],t)*args.content_weight

            loss_s=0
            for gf,sf in zip(g_feats,s_feats):
                g_mean,g_std=calc_mean_std(gf)
                s_mean,s_std=calc_mean_std(sf)
                loss_s+=(mse_loss(g_mean,s_mean)+mse_loss(g_std,s_std))
            loss_s*=args.style_weight

            loss=loss_c+loss_s

            optimizer.zero_grad()
            loss.backward() 

            optimizer.step()

            progress_bar.set_description(f'Loss:{loss.item():4f},Content Loss: {loss_c.item():4f},Style Loss: {loss_s.item():4f}')
            running_loss+=loss.item()
            running_closs+=loss_c.item()
            running_sloss+=loss_s.item()

        scheduler.step()

        running_loss/=len(content_loader)
        running_closs/=len(content_loader)
        running_sloss/=len(content_loader)


        if (epoch+1)% args.log_interval ==0:
            tqdm.write(f'Iter {epoch+1}: Loss{running_loss:4f},Content Loss: {running_closs:4f},Style Loss: {running_sloss:4f}')

        if (epoch+1)% args.save_interval ==0:
            torch.save(decoder.state_dict(),save_dir/f'decoder_epoch_{epoch+1}.pth')
            torch.save(optimizer.state_dict(),save_dir/f'optimizer_epoch_{epoch+1}.pth')

            with torch.no_grad(): 
                output=torch.cat([content_batch,style_batch,g],dim=0)
                save_image(output,save_dir/f'output_epoch_{epoch+1}.jpg',nrow=args.batch_size)

if __name__ == "__main__":
    main()