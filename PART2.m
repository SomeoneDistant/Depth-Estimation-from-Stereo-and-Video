clear;
clc;
addpath('.\Assignment2_A0186492R_YaoYuan\GCMex')
addpath('.\Assignment2_A0186492R_YaoYuan\image')

img_l=double(imread('.\Assignment2_A0186492R_YaoYuan\image\im2.png'));
img_r=double(imread('.\Assignment2_A0186492R_YaoYuan\image\im6.png'));
gt=rgb2gray(im2double(imread('.\Assignment2_A0186492R_YaoYuan\image\depth.png')));
gt=resample(gt',450,892);
gt=resample(gt',375,748);

[H, W, ~] = size(img_l);
N=H*W;

left=reshape(img_l,1,N,3);
right=reshape(img_r,1,N,3);

k=1;
for col = 1:W
    for row = 1:H
        n = (col-1)*H + row;
        if row < H
            i(k)=n;
            j(k)=n+1;
            k=k+1;
        end
        if row > 1
            i(k)=n;
            j(k)=n-1;
            k=k+1;
        end
        if col < W
            i(k)=n;
            j(k)=n+H;
            k=k+1;
        end
        if col > 1
            i(k)=n;
            j(k)=n-H;
            k=k+1;
        end
    end
end
for disparity=1:65
    dis=disparity*H;
    unary(disparity,:)=sum(abs(cat(2,zeros(1,dis,3),right(1,1:N-dis,:))-left),3)./3;
end

[~, segclass] = min(unary); 
segclass = segclass' - 1;

[X, Y] = meshgrid(1:65, 1:65);
labelcost=log(1+((X-Y).^2)./2);

for lambda=1:50
    pairwise=sparse(i,j,lambda);
    [label, ~, ~] = GCMex(segclass, single(unary), pairwise, single(labelcost),1);
    label=reshape(label,H,W);
    result=mat2gray(label);
    if lambda==6
        figure(1)
        imshow(result)
        xlabel('Lambda=6')
        figure(2)
        imshow(gt)
        xlabel('Ground Truth')
    end
    error(lambda)=sum(sum((abs(result-gt))))./N;
end
figure(3)
plot(error)
xlabel('Lambda')
ylabel('Error Rate (L1 Norm)')
