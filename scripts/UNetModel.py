import torch 
import torch.nn as nn 

class U_Net(nn.Module):
    def __init__(self , in_channels=3 , num_classes=19):
        super(U_Net , self).__init__()

        def conv_block(in_channels , out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels , out_channels , kernel_size=3 , padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels , out_channels , kernel_size=3 , padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1=conv_block(in_channels , 64)
        self.enc2=conv_block(64 , 128)
        self.enc3=conv_block(128 , 256)
        self.enc4=conv_block(256 , 512)

        self.pool=nn.MaxPool2d(kernel_size=2 , stride=2)

        ###Decoder###

        self.upconv3 = nn.ConvTranspose2d(512 , 256 , kernel_size=2 , stride=2)
        self.upconv2 = nn.ConvTranspose2d(256 , 128 , kernel_size=2 , stride=2)
        self.upconv1 = nn.ConvTranspose2d(128 , 64 , kernel_size=2 , stride=2)

        self.dec1= conv_block(512 , 256)
        self.dec2= conv_block(256 , 128)
        self.dec3= conv_block(128 , 64)

        self.out=nn.Conv2d(64 , num_classes , kernel_size=1)

    def forward(self , x):
        #Encoder 
        x1=self.enc1(x)  #[B,64,256,512]
        x2=self.enc2(self.pool(x1)) #[B,128,128,256]
        x3=self.enc3(self.pool(x2)) #[B,256,64,128]
        x4=self.enc4(self.pool(x3)) #[B,512,32,64]

        #Decoder
        x=self.upconv3(x4) #[B,256,64,128]
        x=torch.cat((x , x3) , dim=1) #[B,512,64,128]
        x=self.dec1(x) #[B,256,64,128]

        x=self.upconv2(x) #[B,128,128,256]
        x=torch.cat((x, x2) , dim=1) #[B,256,128,256]
        x=self.dec2(x) #[B,128,128,256]

        x=self.upconv1(x) #[B,64,256,512]
        x=torch.cat((x , x1) , dim=1) #[B,128,256,512]
        x=self.dec3(x) #[B,64,256,512]


        return self.out(x) #[B,19,256,512]
