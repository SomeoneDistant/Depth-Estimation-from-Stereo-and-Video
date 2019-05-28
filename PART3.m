clear;
clc;
addpath('.\Assignment2_A0186492R_YaoYuan\GCMex')
addpath('.\Assignment2_A0186492R_YaoYuan\image')

dis_min=0;
step=1e-4;
dis_max=0.01;
dis_num=1+(dis_max-dis_min)./step;
eta=0.05.*(dis_max-dis_min);
epsilon=50;
ws=10./(dis_max-dis_min);
sigma=10;

img_1=double(imread('.\Assignment2_A0186492R_YaoYuan\image\test00.jpg'));
img_2=double(imread('.\Assignment2_A0186492R_YaoYuan\image\test09.jpg'));
[H, W, ~] = size(img_1);
N=H*W;
img_1_prime=reshape(img_1,1,N,3);

cameras=fopen('.\Assignment2_A0186492R_YaoYuan\image\cameras_PART3.txt','r');
camera_mat = fscanf(cameras, '%f %f %f', [3,Inf]);
fclose(cameras);
K_1=camera_mat(:,1:3)';
R_1=camera_mat(:,4:6)';
T_1=camera_mat(:,7);
K_2=camera_mat(:,8:10)';
R_2=camera_mat(:,11:13)';
T_2=camera_mat(:,14);

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
lambda=1./(sqrt(sum((img_1_prime(1,i,:)-img_1_prime(1,j,:)).^2,3))+epsilon);
prior=sparse(i,j,lambda);

nei=4.*ones(H,W);
nei(:,1)=3;
nei(:,end)=3;
nei(1,:)=3;
nei(end,:)=3;
nei(1,1)=2;
nei(1,end)=2;
nei(end,1)=2;
nei(end,end)=2;
nei=reshape(nei,1,N);

u=nei./full(sum(prior));
lambda=ws.*lambda.*u(i);
pairwise=sparse(i,j,lambda);

[label_X, label_Y] = meshgrid(1:dis_num, 1:dis_num);
labelcost=min(step.*abs(label_X-label_Y), eta);

[X,Y]=meshgrid(1:W,1:H);
loc_1=[X(:)';Y(:)';ones(1,N)];
pix_1=impixel(img_1,loc_1(1,:),loc_1(2,:));
for dis=1:dis_num
    loc_2=K_2*R_2'*R_1/K_1*loc_1+(K_2*R_2'*(T_1 - T_2).*step.*(dis-1));
    loc_2=round(loc_2./loc_2(3,:));
    pix_2=impixel(img_2, loc_2(1,:), loc_2(2,:));
    pix_2(isnan(pix_2))=0;
    unary(dis,:)=(sigma./(sigma+sqrt(sum((pix_1-pix_2).^2,2))))';
end
unary=1-(unary./max(unary));

[~,segclass] = min(unary); 
segclass=segclass-1;

[label, ~, ~] = GCMex(segclass, single(unary), pairwise, single(labelcost),1);
label=reshape(label,H,W);
result=mat2gray(label);
imshow(result)
