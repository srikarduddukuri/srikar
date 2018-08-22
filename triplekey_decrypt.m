function final_output=triplekey_decrypt(final_encrypt)

final_encrypt=final_encrypt(:);
%% Computing the Initial Parameter X(i)
s_key='A6C317F6121E96B85B3B';
binary_key=hexToBinaryVector(s_key);
final_key=reshape(binary_key,4,[]);
final_key=final_key';
extract=final_key(7:12,:);
X01=(extract(1,1)*2^0+extract(1,2)*2^1+extract(1,3)*2^2+extract(1,4)*2^3+extract(2,1)*2^4+extract(2,2)*2^5+extract(2,3)*2^6+extract(2,4)*2^7+extract(3,1)*2^8+extract(3,2)*2^9+extract(3,3)*2^10+extract(3,4)*2^11+extract(4,1)*2^12+extract(4,2)*2^13+extract(4,3)*2^14+extract(4,4)*2^15+extract(5,1)*2^16+extract(5,2)*2^17+extract(5,3)*2^18+extract(5,4)*2^19+extract(6,1)*2^20+extract(6,2)*2^21+extract(6,3)*2^22+extract(6,4)*2^23)/(2^24);
X02=(final_key(13,1)+final_key(13,2)+final_key(13,3)+final_key(13,4)+final_key(14,1)+final_key(14,2)+final_key(14,3)+final_key(14,4)+final_key(15,1)+final_key(15,2)+final_key(15,3)+final_key(15,4)+final_key(16,1)+final_key(16,2)+final_key(16,3)+final_key(16,4)+final_key(17,1)+final_key(17,2)+final_key(17,3)+final_key(17,4)+final_key(18,1)+final_key(18,2)+final_key(18,3)+final_key(18,4))/96;
X03=0.9;
mu=3.9999;

%% Generating a Chaotic Sequence
X(1)=(X01+X02+X03)/10;
for i=2:length(final_encrypt)
    X(i)=mu*X(i-1)*(1-X(i-1));
end
Xmin=min(X);
Xmax=max(X);
for i=1:length(X)
    X(i)=floor(((X(i)-Xmin)/Xmax)*255);
end
for i=1:length(final_encrypt)
    final_X{i,1}=dec2binvec(X(i),12);
end
final_X=cell2mat(final_X);

%% Decryption process
for i=1:length(final_encrypt)
   inv_vec{i,1}=dec2binvec(final_encrypt(i),12);
end
inv_vec=cell2mat(inv_vec);
decryption=xor(inv_vec,final_X);
decryption=binvec2dec(decryption);
j=1;
for i=1:3:length(decryption)
    final_decrypt(j,1)=decryption(i);
    j=j+1;
end
final_output=reshape(final_decrypt,[50 50]);
% figure,imshow(final_output,[]);