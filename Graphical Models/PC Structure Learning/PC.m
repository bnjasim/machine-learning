clc
clear all

% Import data
data = load('D1.mat'); 
D = data.D;

A = pc1(D);
disp(A);

A = pc2(D);
disp(A);

A = pc3(D);
disp(A);

% Import data
data = load('D2.mat'); 
D = data.D;

A = pc1(D);
disp(A);

A = pc2(D);
disp(A);

A = pc3(D);
disp(A);

% Import data
data = load('D3.mat'); 
D = data.D;

A = pc1(D);
disp(A);

A = pc2(D);
disp(A);

A = pc3(D);
disp(A);