clc;
clear;
close all;

for j=1:11
    I = im2double(imread(sprintf('%d.png',j)));
    I = imresize(I,0.5);
    imwrite(I,sprintf('%d_small.png',j));
end