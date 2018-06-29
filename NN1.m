load sunspot.dat
year = sunspot(:,1); 
relNums = sunspot(:,2); 
ynrmv = mean(relNums(:)); 
sigy = std(relNums(:)); 
nrmY = relNums;
ymin = min(nrmY(:)); ymax=max(nrmY(:)); 
relNums = 2.0*((nrmY-ymin)/(ymax-ymin)-0.5);
Ss = relNums';
idim = 10;
odim = length(Ss) - idim; 
y = zeros(1, odim);
for i = 1:odim
   y(i) = Ss(i+idim);
   for j = 1:idim
       x(i,idim-j+1)=Ss(i-j+idim);
   end
end
patterns = x'; 
targets = y; 
[numInputs, numPatterns] = size(patterns); 
windowSize = 200;
numWindows = numPatterns - windowSize;
maxEpochs = 100;
numHidden = 5; 
learnRate = 0.001;  
initialWeights1 = 0.5*(rand(numHidden, numInputs)-0.5);
initialWeights2 = 0.5*(rand(1, numHidden)-0.5);
modes = {'CLASSIC', 'FAST HESSIAN'};
numModes = size(modes, 2);
rmses = zeros(numWindows, numModes);
predictions = zeros(numWindows, numModes);
predictionErrors = zeros(numWindows, numModes);
for i = 1:numModes
    mode = modes{:, i};
    isFastHessian = strcmp(mode,'FAST HESSIAN') == 1;
    weights1 = initialWeights1;
    weights2 = initialWeights2;
    for windowNum = 1:numWindows
        windowStart = windowNum;
        windowEnd = windowStart + windowSize - 1;
        windowPatterns = patterns(:, windowStart:windowEnd);
        windowTargets = targets(:, windowStart:windowEnd);
        for epoch = 1:maxEpochs
            % FORWARD PASS 
            hiddenIn = weights1 * windowPatterns;
            hiddenOut = 1.0 ./ (1.0 + exp(-hiddenIn));
            out = weights2 * hiddenOut;
            error = windowTargets - out;
            tss = sum(sum(error.^2));
            rmse = sqrt(0.5*(tss/numPatterns));
            % BACKWARD PASS 
            % From  K to JK.
            % The Jacobian: partial derivatives of the output at K, wrt Wjk. 
            % The gradient: partial derivative of the error at K, wrt Wjk.
            dOkdSk = 1; 
            dSkdWjk = hiddenOut; 
            dOkdWjk = dOkdSk .* dSkdWjk; % jacobian 
            dEkdOk = error;
            dEkdWjk = dEkdOk * dOkdWjk'; % gradient 
            % From K to IJ. 
            % Multiplication matrix for each weight (50 x 200)
            dOjdSj = (hiddenOut .* (1.0 - hiddenOut)); % deriv. of sigmoid
            dSkdSj = (weights2' .* dOjdSj); % Wjk * Oj(1-Oj)
            dSkdSj = repmat(dSkdSj,10,1);
            dSjdWij = [repmat(windowPatterns(1,:),5,1);
                    repmat(windowPatterns(2,:),5,1);
                    repmat(windowPatterns(3,:),5,1);
                    repmat(windowPatterns(4,:),5,1);
                    repmat(windowPatterns(5,:),5,1);
                    repmat(windowPatterns(6,:),5,1);
                    repmat(windowPatterns(7,:),5,1);
                    repmat(windowPatterns(8,:),5,1);
                    repmat(windowPatterns(9,:),5,1);
                    repmat(windowPatterns(10,:),5,1)];
            dOkdWij = dOkdSk .* dSkdSj .* dSjdWij; % jacobian to H layer for each pattern
            dEkdOk = repmat(error, 50, 1);
            dEkdWij = dEkdOk .* dOkdWij; % gradient to H layer for each pattern
            dEkdWij = sum(dEkdWij, 2); % gradient wrt weights to hidden layer (per epoch)
            if isFastHessian
                % combine partials for Wij and Wjk.
                gradient = [dEkdWij', dEkdWjk]; % 55x200
                jacobian = [dOkdWij; dOkdWjk]; % 55x1
                approxHessian = jacobian * jacobian'; % Approximated Hessian
                approxHessian = approxHessian ./ windowSize;
                regularisedHessian = approxHessian + (eye(length(approxHessian)) * 0.001);
                % Newton's for weight deltas
                gradient = gradient ./ windowSize;
                newtons = regularisedHessian\gradient';
                deltaW1 = reshape(newtons(1:50), 5, 10);
                deltaW2 = newtons(51:55)';
            else % CLASSIC mode
                deltaW1 = learnRate * reshape(dEkdWij, 5, 10);
                deltaW2 = learnRate * dEkdWjk;
            end
            weights2 = weights2 + deltaW2;
            weights1 = weights1 + deltaW1;
        end
        rmses(windowNum, i) = rmse;
        predictionPattern = patterns(:,(windowEnd+1));
        predictionTarget = targets(:, (windowEnd+1));
        hiddenIn = weights1 * predictionPattern; 
        hiddenOut = 1.0 ./ (1.0 + exp(-hiddenIn)); 
        prediction = weights2 * hiddenOut;
        predictions(windowNum, i) = prediction;
        predictionError = sqrt(0.5*((prediction - predictionTarget)^2));
        predictionErrors(windowNum, i) = predictionError;
    end
   
end
% PLOTS
% 1) Train Error
xaxis = 1:numWindows;
figure
hold on
for i = 1:numModes
    mode = modes{:, i};
    plot(xaxis, rmses(:,i));
end
hold off
title('Training Error with moving window of size 200')
xlabel('Number of window')
ylabel('Root Mean Squared Error')
% 2) Predictions and Targets.
% Plot 2-3. Blue = prediction; Orange = target.
xaxis = year(211:288);
for i = 1:numModes
    mode = modes{:, i};
    figure
    plot(xaxis, y(201:278), xaxis, predictions(:, i))
    title(strcat('predictions vs targets ', mode))
    ylabel('Sun spots')
end    
% 3) Train and Test Errors. 
% Plot 4-5. Blue = train; Orange = test.
xaxis = 1:numWindows;
for i = 1:numModes
    mode = modes{:, i};
    figure
    plot(xaxis, rmses(:, i), xaxis, predictionErrors(:, i))
    title(strcat('Train and Test Errors : ', mode))
    xlabel('window number')
    ylabel('Root Squared Mean Errors')
end
% Average Train and Test Errors. 
for i = 1:numModes
    mode = modes{:, i};
    fprintf(...
        'Mode: %s\t Average Train Error = %f\t Average Test Error = %f\n',...
            mode, mean(rmses(:,i)), mean(predictionErrors(:,i)))
end