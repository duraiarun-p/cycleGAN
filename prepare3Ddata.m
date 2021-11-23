clc;clear;close all;
%%
mypath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/';
mypathCB='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/db2/trainCB/';
mypathCT='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/db2/trainCT/';


patfolder=dir(mypath);
patfolder(1:2,:) = []; 
bigfolderlen=length(patfolder);

% % bigfolderlen=2;
% for folderi=1:bigfolderlen
% mypath1=[patfolder(folderi).folder,'/',patfolder(folderi).name];
% end
%%
folderi=1;
mypath1=[patfolder(folderi).folder,'/',patfolder(folderi).name];
load(mypath1);
%%
CTcellsiz=size(CTInfoCell);
CTcelllen=zeros(CTcellsiz(1));
for ctcelli=1:CTcellsiz(1)
    CTcelllen(ctcelli)=length(CTInfoCell{1,1});
end
[~,CTcellindex]=max(CTcelllen);
CTInfoCell{2,1}=zeros(100);
CTInfoCell(1:end~= CTcellindex,:)=[];
CT=CTInfoCell{1,3};
%%

CBcellsiz=size(CBCTInfocell);

CBCT=CBCTInfocell{1,5};

patch_size=32;
depth_size=32;
mbSize=patch_size/2;
% 

% CBCT=ones(512,512,88); % For coding purposes not for actual files
%%
% CBblk=cell(1,1);

blkextarctor(CBCT,mypathCB,mbSize,patch_size,depth_size)
blkextarctor(CT,mypathCT,mbSize,patch_size,depth_size)



function blkextarctor(CBCT,mypath,mbSize,patch_size,depth_size)
CBsiz=size(CBCT);
count=1;
for i = 1 : mbSize : CBsiz(1)-mbSize
    for j = 1 : mbSize : CBsiz(2)-mbSize
        for zi=1:mbSize:CBsiz(3)
           if i+patch_size>CBsiz(1)
               i1=CBsiz(1)-patch_size;
           else
               i1=i;
           end
           
           if j+patch_size>CBsiz(2)
               j1=CBsiz(2)-patch_size;
           else
               j1=j;
           end
           
           if zi+depth_size>CBsiz(3)
               zi1=CBsiz(3)-depth_size;
           else
               zi1=zi;
           end
           currentBlk=CBCT(i1:i1+patch_size-1,j1:j1+patch_size-1,zi1:zi1+depth_size-1);
           filepathCB=strcat(mypath,num2str(count),'.tiff');
           
           newtiffwrite(currentBlk,depth_size,filepathCB)
           count=count+1;
            
        end
    end
end
end

function newtiffwrite(currentBlk,depth_size,filepathCB)

for ch=1:depth_size
    imwrite(uint16(currentBlk(:,:,ch)),filepathCB,'Writemode','append');
end


end

