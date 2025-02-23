classdef neuralModel
    %NEURALMODEL Summary of this class goes here
    % dp
    % Github: github.com/DoanCongQui
    % Detailed explanation goes here
    
    properties
        X, W, d, teta, max_epoch
    end
    
    methods
        function self = neuralModel(X, W, d, teta, max_epoch)
            self.X = X;
            self.W = W;
            self.d = d;
            self.teta = teta;
            self.max_epoch = max_epoch;
        end
        
        %{
        @ Function Perceptron
        @ Output is binery
        %}
        function Perceptron(self)
            [m, k] = size(self.X);
            net = zeros(1, k);
            y = zeros(1, k);
            epoch = 0;
            while epoch < self.max_epoch 
                E = 0;
                for i = 1:k
                    net(i) = self.W' * self.X(:,i);
                    if net(i) >= 0
                        y(i) = 1;
                    else
                        y(i) = 0;
                    end    

                    self.W = self.W + self.teta * (self.d(i) - y(i)) * self.X(:,i);
                    
                    E = E + 0.5 * (self.d(i) - y(i))^2;
                    
                    fprintf('\nk = %d\nW =\n', i)
                    disp(self.W)
                    fprintf('E = %.2f\n', E);
                end
                epoch = epoch + 1;
                fprintf('\nEpoch: %d, Error: %f\n-----------------------\n', epoch, E)
                
                if E == 0
                    break
                end
                
            end
        end

        function plot2D(self, x1, x2, tle)
            self.X = self.X(end-1:end, :);
            figure; hold on;
            for i = 1:length(self.d)
                if self.d(i) == 1
                    plot(self.X(1, i), self.X(2, i), 'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b'); 
                else
                    plot(self.X(1, i), self.X(2, i), '^r', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
                end
            end

            % Hien thi bieu do truc quan hoa
            xlabel(x1);
            ylabel(x2);
            title(tle);
            grid on; hold off;
        end
    end
    
end

