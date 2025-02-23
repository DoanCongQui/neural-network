clc;
%{ 
============================
    Du lieu huan luyen
============================
%}

X = [1 1 1 1; 0 0 1 1; 0 1 0 1]; 
W = [0.1; 0.3; 0.5];
d = [0 1 1 1];
n = 0.1;
max_epoch = 10;

%{ 
============================
   Chuong trinh huan luyen
============================
%}

ann = neuralModel(X, W, d, n, max_epoch);
ann.Perceptron()