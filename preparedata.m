clc;clear;close all;
%% old section
% load('/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/APrintOUTImg.mat')
% %%
% 
% CTsiz=size(ACT);
% for ctindex=1:CTsiz(3)
% CT_slice=ACT(:,:,ctindex);
% filepathCT=strcat('/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/trainCT/',num2str(ctindex),'.png');
% imwrite(uint16(CT_slice), filepathCT);
% end
% %%
% CBsiz=size(ACBCT);
% for cbindex=1:CBsiz(3)
% CB_slice=ACBCT(:,:,cbindex);
% filepathCB=strcat('/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/trainCB/',num2str(cbindex),'.png');
% imwrite(uint16(CB_slice), filepathCB);
% end
%%
%% Section to save all slices from all volumes
mypath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/';
% mypathCB='/home/arun/Documents/PyWSPrecision/datasets/printout_2d_data/trainCB/';
% mypathCT='/home/arun/Documents/PyWSPrecision/datasets/printout_2d_data/trainCT/';
% 
mypathCB='/home/arun/Documents/PyWSPrecision/datasets/printout2d/CB/';
mypathCT='/home/arun/Documents/PyWSPrecision/datasets/printout2d/CT/';
% 
patfolder=dir(mypath);
patfolder(1:2,:) = []; 
bigfolderlen=length(patfolder);
% 
% % % bigfolderlen=2;
for folderi=1:bigfolderlen
% % % mypath1=[patfolder(folderi).folder,'/',patfolder(folderi).name];
% % 
% %
% folderi=6;
mypath1=[patfolder(folderi).folder,'/',patfolder(folderi).name];
load(mypath1);
% %%
CTcellsiz=size(CTInfoCell);
CTcelllen=zeros(CTcellsiz(1),1);
for ctcelli=1:CTcellsiz(1)
    CTcelllen(ctcelli)=length(CTInfoCell{ctcelli,1});
end
[~,CTcellindex]=max(CTcelllen);
% CTInfoCell{2,1}=zeros(100);
CTInfoCell(1:end~= CTcellindex,:)=[];
CT=CTInfoCell{1,3};
CTsize=size(CT);
% ctslice=1;
for ctslice=1:CTsize(3)
     CT_slice=CT(:,:,ctslice);
    filepathCT=strcat(mypathCT,'CT-',num2str(folderi),'-','Slice-',num2str(ctslice),'.png');
    imwrite(uint16(CT_slice), filepathCT);
end
%%
CBcellsiz=size(CBCTInfocell);
% cbcelli=2;
for cbcelli=1:CBcellsiz(1)
CBCT=CBCTInfocell{cbcelli,5}; % Choosing resampled CBCT (height and width, not the depth)
CBsize=size(CBCT);
% cbslice=1;
    for cbslice=1:CBsize(3)
        CB_slice=CT(:,:,cbslice);
        filepathCB=strcat(mypathCB,'CB-',num2str(folderi),'-Ser-',num2str(cbcelli),'-Slice-',num2str(cbslice),'.png');
        imwrite(uint16(CB_slice), filepathCB);
    end
end
disp(folderi)
end
%%
% % %%
mypath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/';
mypathCB='/home/arun/Documents/PyWSPrecision/datasets/printout2d/CB/';
mypathCT='/home/arun/Documents/PyWSPrecision/datasets/printout2d/CT/';

CBlis=dir(mypathCB);
CTlis=dir(mypathCT);
CBlis(1:2)=[];
CTlis(1:2)=[];
CTlislen=length(CTlis);
CBlislen=length(CBlis);
trainlen=round(0.7*min([CBlislen,CTlislen]));
validlen=round(0.2*min([CBlislen,CTlislen]));
testlen=round(0.1*min([CBlislen,CTlislen]));
%%

%%
mypathtestCT='/home/arun/Documents/PyWSPrecision/datasets/printout2d/testCT/';
mypathvalidCT='/home/arun/Documents/PyWSPrecision/datasets/printout2d/validCT/';
mypathtrainCT='/home/arun/Documents/PyWSPrecision/datasets/printout2d/trainCT/';
%%
writeslices(CTlis,CTlislen,trainlen,validlen,testlen,mypathtestCT,mypathvalidCT,mypathtrainCT);
%%
mypathtestCB='/home/arun/Documents/PyWSPrecision/datasets/printout2d/testCB/';
mypathvalidCB='/home/arun/Documents/PyWSPrecision/datasets/printout2d/validCB/';
mypathtrainCB='/home/arun/Documents/PyWSPrecision/datasets/printout2d/trainCB/';
%%
writeslices(CBlis,CBlislen,trainlen,validlen,testlen,mypathtestCB,mypathvalidCB,mypathtrainCB);
%%
function writeslices(CTlis,CTlislen,trainlen,validlen,testlen,mypathtestCT,mypathvalidCT,mypathtrainCT)
CTtrainindex=randperm(CTlislen,trainlen);
CTtrainlis=CTlis(CTtrainindex);
CTvalidlis=CTlis;
CTvalidlis(CTtrainindex)=[];
% CTvalidlislen=length(CTvalidlis);

CTtestlis=CTvalidlis;
CTtestindex=randperm(validlen,testlen);
CTtestlis=CTtestlis(CTtestindex);

CTvalidlis(CTtestindex)=[];
CTvalidlis(validlen:end)=[];
CTvalidlislen=length(CTvalidlis);

%%
for testi=1:testlen
    sourcefile=strcat(CTtestlis(testi).folder,'/',CTtestlis(testi).name);
    copyfile(sourcefile,mypathtestCT);
end

for validi=1:CTvalidlislen
    sourcefile=strcat(CTvalidlis(validi).folder,'/',CTvalidlis(validi).name);
    copyfile(sourcefile,mypathvalidCT);
end
for traini=1:trainlen
    sourcefile=strcat(CTtrainlis(traini).folder,'/',CTtrainlis(traini).name);
    copyfile(sourcefile,mypathtrainCT);
end
end