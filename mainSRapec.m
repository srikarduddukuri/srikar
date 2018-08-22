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
I = imread('Extracted Frames\Image2.jpg');
% Resizing the image
I = imresize(I,[256 256]);
figure,imshow(I,'Border','loose');
title('Input Image');
% Finding the size of the image
[m n o] = size(I);
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
fin=imread('fingerprint.jpg');
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
