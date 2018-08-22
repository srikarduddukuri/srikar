function final_seg=segmentation_back(I,m,n)
R=im2double(I(:,:,1));
G=im2double(I(:,:,2));
B=im2double(I(:,:,3));
mask=imread('mask1.bmp');
for i=1:m
    for j=1:n
        if mask(i,j) ~= 0
            im1(i,j)=R(i,j);
            im2(i,j)=G(i,j);
            im3(i,j)=B(i,j);
        else
            im1(i,j)=1;
            im2(i,j)=1;
            im3(i,j)=1;
        end
    end
end         

final_seg=cat(3,im1,im2,im3);