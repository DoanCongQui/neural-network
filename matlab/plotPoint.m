classdef plotPoint
    %PLOTPOINT Summary of this class goes here
    % dp
    % Github: github.com/DoanCongQui
    % Detailed explanation goes here
    
    properties(Access = public)
        X, d
    end
    
    methods
        function obj = plotPoint(X, d)
            obj.X = X;
            obj.d = d;
            
        end
        
        function plot2D(obj, x1, x2, tle)
            figure; hold on;
            for i = 1:length(obj.d)
                if obj.d(i) == 1
                    plot(obj.X(1, i), obj.X(2, i), 'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b'); 
                else
                    plot(obj.X(1, i), obj.X(2, i), '^r', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
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

