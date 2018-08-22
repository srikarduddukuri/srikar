clc;
close all;
clear all;
%% video to frame converter
delete('Extracted Frames\*.jpg');
[p,f] = uigetfile('*.avi;*.mpeg;*.mp4');
vid = VideoReader([f,p]);
numFrames = vid.NumberOfFrames;
n = numFrames;
 for i = 1:n
 frames = read(vid,i);
%  frames=imcrop(frames,[332 6 308 278]);
 imwrite(frames,['Extracted Frames\Image' num2str(i),'.jpg']);
%  im(i)=image(frames);
 end
%% Read input data
[f,p]=uigetfile('*.jpg');
I = imread([p,f]);
% Resizing the image
I = imresize(I,[256 256]);
figure,imshow(I,'Border','loose');
title('Input Image');
% Finding the size of the image
[m n o] = size(I);
I1=rgb2gray(I);
% Segmenting the backgroundo
%% segmentation
% imageSegmenter
% initialize
% draw free hand
% Accept
% Show Binary
% Export ---> Final Segmentation
% imwrite(BW,'mask.bmp');
final_seg=segmentation_back(I,m,n);
figure,imshow(final_seg);
title('Background Removed image');
% final_seg=I;
%% Read the biometric signal
fin=imread('finger.bmp');
[k l z]=size(fin);
% Converting the image to grayscale
if z==3
    gr=double(im2uint8(rgb2gray(fin)));
else
    gr=double(im2uint8(fin));
end
gr=imresize(gr,[50 50]);
figure,imshow(gr,[]);
title('Input Biometric signal');
a=50;
b=50;
%% Triple key Encryption
encrypt_image=triplekey_encrypt(gr);
figure,imshow(encrypt_image,[]);
title('ENCRYPTED IMAGE');
encrypt_image1=uint8(encrypt_image);

% Vectorizing the encrypted data
vect=encrypt_image(:)';
% Sorting the vectorized data
[vect_desend,position]=sort(vect,'descend');


%% QSWT Estimation, Data hiding and Encoding using SPIHT
t = 1;
% DWT R region
[LL1_1,LH1_1,HL1_1,HH1_1]=dwt2(final_seg(:,:,1),'sym4');
[LL2_1,LH2_1,HL2_1,HH2_1]=dwt2(LL1_1,'sym4');
[LL1_1,LH1_1,HL1_1,HH1_1,LL2_1,LH2_1,HL2_1,HH2_1,pos1,t,TT] = qswt(LL1_1,LH1_1,HL1_1,HH1_1,LL2_1,LH2_1,HL2_1,HH2_1,vect_desend,a,b,t);

% DWT G region
[LL1_2,LH1_2,HL1_2,HH1_2]=dwt2(final_seg(:,:,2),'sym4');
[LL2_2,LH2_2,HL2_2,HH2_2]=dwt2(LL1_2,'sym4');
[LL1_2,LH1_2,HL1_2,HH1_2,LL2_2,LH2_2,HL2_2,HH2_2,pos2] = qswt(LL1_2,LH1_2,HL1_2,HH1_2,LL2_2,LH2_2,HL2_2,HH2_2,vect_desend,a,b,t);

% DWT B region
[LL1_3,LH1_3,HL1_3,HH1_3]=dwt2(final_seg(:,:,3),'sym4');
[LL2_3,LH2_3,HL2_3,HH2_3]=dwt2(LL1_3,'sym4');
[LL1_3,LH1_3,HL1_3,HH1_3,LL2_3,LH2_3,HL2_3,HH2_3,pos3] = qswt(LL1_3,LH1_3,HL1_3,HH1_3,LL2_3,LH2_3,HL2_3,HH2_3,vect_desend,a,b,t);

%% 

R1=idwt2(LL2_1,LH2_1,HL2_1,HH2_1,'sym4');
R2=idwt2(LL1_1,LH1_1,HL1_1,HH1_1,'sym4');

G1=idwt2(LL2_2,LH2_2,HL2_2,HH2_2,'sym4');
G2=idwt2(LL1_2,LH1_2,HL1_2,HH1_2,'sym4');

B1=idwt2(LL2_3,LH2_3,HL2_3,HH2_3,'sym4');
B2=idwt2(LL1_3,LH1_3,HL1_3,HH1_3,'sym4');

% Fusion of R,G,B 
concat_im=cat(3,R2,G2,B2);
figure,imshow(concat_im);
imwrite(concat_im,'data_embb.jpg');
title('Data Embedded Image');
clc;
concat_im=im2double(imread('data_embb.jpg'));
%% JPEG Compression
input_image_128x128_1 = concat_im(:,:,1);
input_image_128x128_2 = concat_im(:,:,2);
input_image_128x128_3 = concat_im(:,:,3);

dct_8x8_image_of_128x128_1 = image_8x8_block_dct(input_image_128x128_1);
dct_8x8_image_of_128x128_2 = image_8x8_block_dct(input_image_128x128_2);
dct_8x8_image_of_128x128_3 = image_8x8_block_dct(input_image_128x128_3);

mean_matrix_8x8_1 = zeros( 8,8 );
mean_matrix_8x8_2 = zeros( 8,8 );
mean_matrix_8x8_3 = zeros( 8,8 );


    
    % in each picture loop over 8x8 elements (128x128 = 256 * 8x8 elements)
   for m = 0:15
      for n = 0:15
         mean_matrix_8x8_1 = mean_matrix_8x8_1 + abs( dct_8x8_image_of_128x128_1(m*8+[1:8],n*8+[1:8]) ).^2;
         mean_matrix_8x8_2 = mean_matrix_8x8_2 + abs( dct_8x8_image_of_128x128_2(m*8+[1:8],n*8+[1:8]) ).^2;
         mean_matrix_8x8_3 = mean_matrix_8x8_3 + abs( dct_8x8_image_of_128x128_3(m*8+[1:8],n*8+[1:8]) ).^2;
      end
   end


mean_matrix_8x8_transposed_1 = mean_matrix_8x8_1';
mean_matrix_8x8_transposed_2 = mean_matrix_8x8_2';
mean_matrix_8x8_transposed_3 = mean_matrix_8x8_3';

mean_vector_1 = mean_matrix_8x8_transposed_1(:);
mean_vector_2 = mean_matrix_8x8_transposed_2(:);
mean_vector_3 = mean_matrix_8x8_transposed_3(:);

[sorted_mean_vector_1,original_indices_1] = sort( mean_vector_1 );
[sorted_mean_vector_2,original_indices_2] = sort( mean_vector_2 );
[sorted_mean_vector_3,original_indices_3] = sort( mean_vector_3 );

sorted_mean_vector_1 = sorted_mean_vector_1(end:-1:1);
original_indices_1 = original_indices_1(end:-1:1);
sorted_mean_vector_2 = sorted_mean_vector_2(end:-1:1);
original_indices_2 = original_indices_2(end:-1:1);
sorted_mean_vector_3 = sorted_mean_vector_3(end:-1:1);
original_indices_3 = original_indices_3(end:-1:1);

coef_selection_matrix_1 = zeros(8,8);
coef_selection_matrix_2 = zeros(8,8);
coef_selection_matrix_3 = zeros(8,8);

compressed_set = [1 3 5 10 15 20 30 40];
% compression
for number_of_coefficient = 1:64
    
    [y1,x1] = find(mean_matrix_8x8_1==max(max(mean_matrix_8x8_1)));
    [y2,x2] = find(mean_matrix_8x8_2==max(max(mean_matrix_8x8_2)));
    [y3,x3] = find(mean_matrix_8x8_3==max(max(mean_matrix_8x8_3)));
    
    coef_selection_matrix_1(y1,x1) = 1;
    coef_selection_matrix_2(y2,x2) = 1;
    coef_selection_matrix_3(y3,x3) = 1;
    
    selection_matrix_1 = repmat( coef_selection_matrix_1,32,32 );
    selection_matrix_2 = repmat( coef_selection_matrix_2,32,32 );
    selection_matrix_3 = repmat( coef_selection_matrix_3,32,32 );
    
    mean_matrix_8x8_1(y1,x1) = 0;
    mean_matrix_8x8_2(y2,x2) = 0;
    mean_matrix_8x8_3(y3,x3) = 0;
    
    compressed_image_1 = image_8x8_block_dct(concat_im(:,:,1)) .* selection_matrix_1;
    compressed_image_2 = image_8x8_block_dct(concat_im(:,:,2)) .* selection_matrix_2;
    compressed_image_3 = image_8x8_block_dct(concat_im(:,:,3)) .* selection_matrix_3;
end



% decompression
for number_of_coefficient = 1:64
    
    restored_image_1 = image_8x8_block_inv_dct( compressed_image_1 );
    restored_image_2 = image_8x8_block_inv_dct( compressed_image_2 );
    restored_image_3 = image_8x8_block_inv_dct( compressed_image_3 );
    
   
end
    
    
%%

concat_im1=cat(3,restored_image_1,restored_image_2,restored_image_3);
figure,imshow(concat_im1);
title('compressed Image');
imwrite(concat_im1,'data_embb1.jpg');

concat_im1=im2double(imread('data_embb1.jpg'));
%% PSNR calculation
% psnr1=psnr(concat_im1(:,:,1),R2);
% psnr2=psnr(concat_im1(:,:,2),G2);
% psnr3=psnr(concat_im1(:,:,3),B2);
% 
% PSNR=mean([psnr1,psnr2,psnr3])

%% Decryption

[LL3_1,LH3_1,HL3_1,HH3_1]=dwt2(concat_im1(:,:,1),'sym4');
[LL4_1,LH4_1,HL4_1,HH4_1]=dwt2(LL3_1,'sym4');

[LL3_2,LH3_2,HL3_2,HH3_2]=dwt2(concat_im1(:,:,2),'sym4');
[LL4_2,LH4_2,HL4_2,HH4_2]=dwt2(LL3_2,'sym4');

[LL3_3,LH3_3,HL3_3,HH3_3]=dwt2(concat_im1(:,:,3),'sym4');
[LL4_3,LH4_3,HL4_3,HH4_3]=dwt2(LL3_3,'sym4');
[p q]=size(LH3_1);


TTx=TT(2:end,:);
[x y]=size(TT);
final=LH1_1;
xx=1;
for ii=1:x
        for jj=1:y-1
            if xx<=2500
            final(TTx(xx,1),TTx(xx,2))=LH3_1(TTx(xx,1),TTx(xx,2))./(1+0.5.*TTx(xx,3));
            xx=xx+1;
            end
        end
end
if pos1 == 1
      W1=(LH3_1-LH1_1)./(LH1_1.*0.5);
      LH3_2=LH3_2-LH1_2;
      LH3_3=LH3_3-LH1_1;
elseif pos1 == 2
    HL3_1=HL3_1-HL1_1;
    HL3_2=HL3_2-HL1_2;
    HL3_3=HL3_3-HL1_1;
elseif pos1 == 3
    HH3_1=HH3_1-HH1_1;
    HH3_2=HH3_2-HH1_2;
    HH3_3=HH3_3-HH1_1;
end
% Inverse DWT
R3=idwt2(LL3_1,LH3_1,HL3_1,HH3_1,'sym4');

G3=idwt2(LL3_2,LH3_2,HL3_2,HH3_2,'sym4');

B3=idwt2(LL3_3,LH3_3,HL3_3,HH3_3,'sym4');

concat_im=cat(3,R3,G3,B3);
figure,imshow(concat_im);
title('Recovered Image');

%% Reconstruction

%      recon=[position' vect_desend'];
position1=position';
trans=TTx(:,3);
final_embed=zeros(a,b);
for i=1:a*b
    final_embed(position1(i,1))=trans(i);
end
figure,imshow(final_embed,[]);
title('Recovered Encrypted Image');
final_embed1=uint8(final_embed);
decrypt_image=triplekey_decrypt(final_embed);
figure,
imshow(decrypt_image,[]);
title('Recovered Image');
imwrite(decrypt_image,'decrypt.bmp');

%% Performance metrices
% Histogram comparission
figure,subplot(1,2,1),imhist(encrypt_image1);
title('Input biometric Image Histogram');
subplot(1,2,2),imhist(final_embed1);
title('Recovered Encrypted Image Histogram');

% Finding the correlation between the images
Correlation=corr2(encrypt_image1,final_embed1);
gr1=rgb2gray(imresize(imread('fingerprint.jpg'),[50 50]));
Correlation1=corr2(gr1,decrypt_image);

% Finding the entropy of input encrypted image and recovered encrypted
% image
entropy1=entropy(encrypt_image1)
entropy2=entropy(final_embed1)

% Finding the compression ratio
c=imfinfo('data_embb1.jpg');
ib=c.Width*c.Height*c.BitDepth/8;
cb=c.FileSize;
compression_ratio=(ib/cb)/100

if Correlation1 > 0.8
    disp('Person is Authentic');
    msgbox('Person is Authentic');
else
    disp('Unauthorised Person');
    msgbox('Unauthorised Person');
end
%correlation Distribution
ii =1;
for i =1:size(gr,1)-1
    for j = 1:size(gr,2)-1
        horizontal{ii,1} = [gr(i,j) gr(i+1,j)];
        vertical{ii,1} = [gr(i,j) gr(i,j+1)];
        diagonal{ii,1} = [gr(i,j) gr(i+1,j+1)];
        ii = ii+1;
    end
end

horizontal = cell2mat(horizontal);
vertical = cell2mat(vertical);
diagonal = cell2mat(diagonal);

figure,
subplot(3,1,1),plot(horizontal(:,1),horizontal(:,2),'.');
title('Plain image Correlation distributions Horizontal')
xlabel('P(x,y)');
ylabel('P(x+1,y)');
axis([0 250 0 250]);
subplot(3,1,2),
plot(vertical(:,1),vertical(:,2),'.');
title('Plain image Correlation distributions Vertical');
xlabel('P(x,y)');
ylabel('P(x,y+1)');
axis([0 250 0 250]);
subplot(3,1,3),plot(diagonal(:,1),diagonal(:,2),'.');
title('Plain image Correlation distributions Diagonal')
xlabel('P(x,y)');
ylabel('P(x+1,y+1)');
axis([0 250 0 250]);
ii =1;
for i =1:size(encrypt_image,1)-1
    for j = 1:size(encrypt_image,2)-1
        horizontal1{ii,1} = [encrypt_image(i,j) encrypt_image(i+1,j)];
        vertical1{ii,1} = [encrypt_image(i,j) encrypt_image(i,j+1)];
        diagonal1{ii,1} = [encrypt_image(i,j) encrypt_image(i+1,j+1)];
        ii = ii+1;
    end
end
horizontal1 = cell2mat(horizontal1);
vertical1 = cell2mat(vertical1);
diagonal1 = cell2mat(diagonal1);
figure,
subplot(3,1,1),
plot(horizontal1(:,1),horizontal1(:,2),'.');
title('Cipher image Correlation distributions Horizontal')
xlabel('P(x,y)');
ylabel('P(x+1,y)');
axis([0 250 0 250]);
subplot(3,1,2),
plot(vertical1(:,1),vertical1(:,2),'.');
title('Cipher image Correlation distributions Vertical');
xlabel('P(x,y)');
ylabel('P(x,y+1)');
axis([0 250 0 250]);
subplot(3,1,3),plot(diagonal1(:,1),diagonal1(:,2),'.');
title('Cipher image Correlation distributions Diagonal')
xlabel('P(x,y)');
ylabel('P(x+1,y+1)');
axis([0 250 0 250]);


