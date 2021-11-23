clc;clear;close all;
%%
% load('/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/30102021/alpha/run26/checkpoints/LearningRateScheduler_Result.mat')
load('/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/30102021/alpha/run56/checkpoints/LearningRateScheduler_Result.mat')
%%
train_resN=train_res;test_resN=test_res;
%%
for chi=1:4
% train_resN(:,chi)=train_resN(:,chi)/max(train_resN(:,chi));
% test_resN(:,chi)=test_resN(:,chi)/max(test_resN(:,chi));
train_resN(:,chi)=rescale(train_resN(:,chi));
test_resN(:,chi)=rescale(test_resN(:,chi));
end
%%
stop=30;
figure(1);
subplot(121),
semilogx(lrs,train_resN(:,1));hold on;
semilogx(lrs,train_resN(:,2));
semilogx(lrs,test_resN(:,1));
semilogx(lrs,test_resN(:,2));hold off;
title('full learning rate Generator loss');
legend({'Tr1','Tr2','Tt1','Tt2'});
% subplot(122),
% semilogx(lrs(1:stop),train_resN(1:stop,1));hold on;
% semilogx(lrs(1:stop),train_resN(1:stop,2));
% semilogx(lrs(1:stop),test_resN(1:stop,1));
% semilogx(lrs(1:stop),test_resN(1:stop,2));hold off;
% title('required learning rate range');
% legend({'Tr1','Tr2','Tt1','Tt2'});
%%
figure(2);
subplot(121),
semilogx(lrs,train_resN(:,3));hold on;
semilogx(lrs,train_resN(:,4));
semilogx(lrs,test_resN(:,3));
semilogx(lrs,test_resN(:,4));hold off;
title('full learning rate Discriminator loss');
legend({'Tr1','Tr2','Tt1','Tt2'});
% subplot(122),
% semilogx(lrs(1:stop),train_resN(1:stop,3));hold on;
% semilogx(lrs(1:stop),train_resN(1:stop,4));
% semilogx(lrs(1:stop),test_resN(1:stop,3));
% semilogx(lrs(1:stop),test_resN(1:stop,4));hold off;
% title('required learning rate range');
% legend({'Tr1','Tr2','Tt1','Tt2'});
%%
figure(3);
% subplot(121),
stop=10;
semilogx(lrs(1:stop),train_res(1:stop));hold on;
semilogx(lrs(1:stop),test_res(1:stop));hold off;
title('full learning rate Discriminator loss');
legend({'Tr1','Tt1'});