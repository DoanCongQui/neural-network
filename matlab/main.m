clc;
e = 0;
n = 0.1; 
tam = 0; 
W = [0.1; 0.3; 0.5];
X = [1 1 1 1; 0 0 1 1; 0 1 0 1]; 
d = [0 0 0 1];

ann = neuralModel(X, W, d, n, 10);
plotP = plotPoint(X(end-1:end, :), d);

ann.Perceptron()
plotP.plot2D('Bien do 475Hz', 'Bien do 555Hz', 'Chat luong vien gach')
