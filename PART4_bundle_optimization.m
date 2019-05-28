clear;
clc;
addpath('.\Assignment2_A0186492R_YaoYuan\GCMex')
addpath('.\Assignment2_A0186492R_YaoYuan\Road\src')
addpath('.\Assignment2_A0186492R_YaoYuan\Road')
addpath('.\Assignment2_A0186492R_YaoYuan\PART4_initialization')
addpath('.\Assignment2_A0186492R_YaoYuan\PART4_bundle_optimization')

dis_min=0;
step=1e-4;
dis_max=1e-2;
dis_num=1+(dis_max-dis_min)./step;
eta=0.05.*(dis_max-dis_min);
epsilon=50;
ws=10./(dis_max-dis_min);
sigmac=10;
sigmad=2.5;
nei_num=3;
iter_num=2;

img=double(imread('.\Assignment2_A0186492R_YaoYuan\Road\src\test0000.jpg'));
[H,W,~]=size(img);
N=H*W;

cameras=fopen('.\Assignment2_A0186492R_YaoYuan\Road\cameras.txt','r');
camera_mat=fscanf(cameras,'%f %f %f',[3,Inf]);
fclose(cameras);

k=1;
for col=1:W
    for row=1:H
        n=(col-1)*H + row;
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

[label_X,label_Y]=meshgrid(1:dis_num,1:dis_num);
labelcost=min(step.*abs(label_X-label_Y), eta);

[X,Y]=meshgrid(1:W,1:H);
loc_cen=[X(:)';Y(:)';ones(1,N)];

for iter=1:iter_num
    if iter==1
        file_root='.\Assignment2_A0186492R_YaoYuan\PART4_initialization';
        nei_start=nei_num*2;
    else
        file_root='.\Assignment2_A0186492R_YaoYuan\PART4_bundle_optimization\Iter_1';
        nei_start=nei_num*3;
    end
    for n=nei_start:140-nei_start
        img_cen=double(imread(['.\Assignment2_A0186492R_YaoYuan\Road\src\test',sprintf('%04d',n),'.jpg']));
        img_cen_prime=reshape(img_cen,1,N,3);
        
        seq=n*7;
        K_cen=camera_mat(:,1+seq:3+seq)';
        R_cen=camera_mat(:,4+seq:6+seq)';
        T_cen=camera_mat(:,7+seq);
        
        lambda=1./(sqrt(sum((img_cen_prime(1,i,:)-img_cen_prime(1,j,:)).^2,3))+epsilon);
        prior=sparse(i,j,lambda);
        u=nei./full(sum(prior));
        lambda=ws.*lambda.*u(i);
        pairwise=sparse(i,j,lambda);
        
        L_total=zeros(dis_num,N);
        for b=[n-3,n-2,n-1,n+1,n+2,n+3]
        img_nei=double(imread(['.\Assignment2_A0186492R_YaoYuan\Road\src\test',sprintf('%04d',b),'.jpg']));
        dis_nei=getfield(load([file_root,'\test',sprintf('%04d',b),'.mat']),'label');
        
        seq=b*7;
        K_nei=camera_mat(:,1+seq:3+seq)';
        R_nei=camera_mat(:,4+seq:6+seq)';
        T_nei=camera_mat(:,7+seq);

        pix_cen=impixel(img_cen,loc_cen(1,:),loc_cen(2,:));
            for dis=1:dis_num
                loc_nei=K_nei*R_nei'*R_cen/K_cen*loc_cen+K_nei*R_nei'*(T_cen-T_nei).*step.*(dis-1);
                loc_nei=round(loc_nei./loc_nei(3,:));
                pix_nei=impixel(img_nei, loc_nei(1,:), loc_nei(2,:));
                pix_nei(isnan(pix_nei))=0;
                pc=(sigmac./(sigmac+sqrt(sum((pix_cen-pix_nei).^2,2))))';
    
                loc_nei=[min(max(loc_nei(1,:),1),W);min(max(loc_nei(2,:),1),H);ones(1,N)];
                trans_loc=sub2ind([H,W],loc_nei(2,:),loc_nei(1,:));
                
                loc_n2c=K_cen*R_cen'*R_nei/K_nei*loc_nei+K_cen*R_cen'*(T_nei-T_cen).*(dis_nei(trans_loc)-1).*step;
                loc_n2c=round(loc_n2c./loc_n2c(3,:));
                
                pd=exp(-sum((loc_n2c(1:2,:)-loc_cen(1:2,:)).^2)./(2.*sigmad.^2));
                
                L(dis,:)=pc.*pd;
            end
            L_total=L_total+L;
        end
        unary=1-L_total./max(L_total);
        [~,segclass] = min(unary); 
        segclass=segclass-1;
        [label,~,~]=GCMex(segclass,single(unary),pairwise,single(labelcost),1);
        label=reshape(label,H,W);
        result=mat2gray(label);
        imwrite(result,['.\Assignment2_A0186492R_YaoYuan\PART4_bundle_optimization\Iter_',num2str(iter),'\test',sprintf('%04d',n),'.jpg']);
        save(['.\Assignment2_A0186492R_YaoYuan\PART4_bundle_optimization\Iter_',num2str(iter),'\test',sprintf('%04d',n),'.mat'],'label');
    end
end
