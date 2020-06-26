import torch
import torch.nn as nn

def conv_layers(input_f,output_f):
    conv=nn.Sequential(
        nn.Conv2d(input_f,output_f,kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_f,output_f,kernel_size=3),
        nn.ReLU(inplace=True)        
    )
    return conv
def crop(tensor,target):
    target_size=target.size()[2]
    tensor_size=tensor.size()[2]
    diff=tensor_size-target_size
    diff=diff//2
    return tensor[:,:,diff:tensor_size-diff,diff:tensor_size-diff]
    

class UNET(nn.Module):
    def __init__(self):
        super(UNET,self).__init__()
        
        self.maxpool2d=nn.MaxPool2d(kernel_size=2,stride=2)
        self.down1=conv_layers(1,64)
        self.down2=conv_layers(64,128)
        self.down3=conv_layers(128,256)
        self.down4=conv_layers(256,512)
        self.down5=conv_layers(512,1024)
        
        self.uptrans1=nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=2,stride=2)
        self.upconv2=conv_layers(1024,512)
        self.uptrans3=nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=2,stride=2)
        self.upconv4=conv_layers(512,256)
        self.uptrans5=nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2)
        self.upconv6=conv_layers(256,128)
        self.uptrans7=nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2)
        self.upconv8=conv_layers(128,64)
        
        self.output=nn.Conv2d(in_channels=64,out_channels=2,kernel_size=1)
        
    def forward(self,image):
        
        #first part of the down convolution
        x1=self.down1(image)  #feeding in the image
        print(x1.size)
        x2=self.maxpool2d(x1)
        x3=self.down2(x2)
        print("  ",x1.size())
        x4=self.maxpool2d(x3)
        x5=self.down3(x4)
        print("    ",x1.size())
        x6=self.maxpool2d(x5)
        x7=self.down4(x6)
        print("      ",x1.size())
        x8=self.maxpool2d(x7)
        x9=self.down5(x8)
        print("        ",x9.size())
        
        xt=self.uptrans1(x9)
        y=crop(x7,xt)
        x_up=self.upconv2(torch.cat([xt,y],1))
        print("      ",x_up.size())
        
        xt=self.uptrans3(x_up)
        y=crop(x5,xt)
        x_up=self.upconv4(torch.cat([xt,y],1))
        print("    ",x_up.size())
        
        xt=self.uptrans5(x_up)
        y=crop(x3,xt)
        x_up=self.upconv6(torch.cat([xt,y],1))
        print("  ",x_up.size())
        
        xt=self.uptrans7(x_up)
        y=crop(x1,xt)
        x_up=self.upconv8(torch.cat([xt,y],1))
        print(x_up.size())
        
        
if __name__=="__main__":
    image=torch.rand((1,1,572,572))
    model=UNET()
    #print(model(image))