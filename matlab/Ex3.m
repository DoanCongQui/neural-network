clc;

%{ 
============================
    Du lieu huan luyen
============================
- Cho teta=0.1
- W = [0.3 0.5]
- phi = 0.1
- Max epoch = 100
%}
n = 0.1; 
W = [0.1; 0.3; 0.5];
X = [1 1 1 1 1 1 1 1; 0.958 1.043 1.907 0.780 0.579 0.003 0.001 0.014; 0.003 0.001 0.003 0.002 0.001 0.105 1.748 1.839];
d = [1 1 1 1 1 0 0 0];
max_epoch = 10;

%{ 
============================
   Chuong trinh huan luyen
============================
%}

ann = neuralModel(X, W, d, n, max_epoch);

ann.Perceptron()


%{ 
============================
    Truc quan du lieu
============================
%}
ann.plot2D('Bien do 475Hz', 'Bien do 555Hz', 'Chat luong vien gach')


