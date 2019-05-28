clear;
clc;
addpath('.\Assignment2_A0186492R_YaoYuan\GCMex')
addpath('.\Assignment2_A0186492R_YaoYuan\image')

img=double(imread('.\Assignment2_A0186492R_YaoYuan\image\bayes_in.jpg'));
img_gt=double(imread('.\Assignment2_A0186492R_YaoYuan\image\bayes_out.jpg'));

[H,W,~]=size(img);
N=H*W;

foreg=[0,0,255];
backg=[245,210,110];

dist_foreg=(abs(img(:,:,1)-foreg(1))+abs(img(:,:,2)-foreg(2))+abs(img(:,:,3)-foreg(3)))./3;
dist_backg=(abs(img(:,:,1)-backg(1))+abs(img(:,:,2)-backg(2))+abs(img(:,:,3)-backg(3)))./3;
unary=[reshape(dist_foreg,1,N);reshape(dist_backg,1,N)];
segclass=double(unary(1,:)>=unary(2,:));

k=1;
for col=1:W
    for row=1:H
        n=(col-1)*H+row;
        if row<H
            i(k)=n;
            j(k)=n+1;
            k=k+1;
        end
        if row>1
            i(k)=n;
            j(k)=n-1;
            k=k+1;
        end
        if col<W
            i(k)=n;
            j(k)=n+H;
            k=k+1;
        end
        if col>1
            i(k)=n;
            j(k)=n-H;
            k=k+1;
        end
    end
end

labelcost=[0,1;1,0];

for lambda=1:200
    pairwise=sparse(i,j,lambda);
    [label,~,~]=GCMex(segclass,single(unary),pairwise,single(labelcost),0);
    label=reshape(label,H,W);
    label1=label*backg(1)+(1-label)*foreg(1);
    label2=label*backg(2)+(1-label)*foreg(2);
    label3=label*backg(3)+(1-label)*foreg(3);
    result=cat(3,label1,label2,label3);
    if lambda==178
        figure(1)
        imshow(uint8(result))
        xlabel('Lambda=178')
        figure(2)
        imshow(img_gt)
        xlabel('Ground Truth')
    end
    error(lambda)=sum(sum(sum(abs(result-img_gt),3)./3))./N;
end
figure(3)
plot(error)
xlabel('Lambda')
ylabel('Error Rate (L1 Norm)')
