clear; clc; close all;
% 设定四个分布的参数
% 指数分布
lambda1 = 0.01;          % 失效率常数
% 瑞利分布
lambda2 = 0.0001;          % 尺度参数
% 威布尔分布
gama = 3;            % 形状参数  
ceita  = 300;         % 尺度参数 
% 对数正态分布
mu    = 3;           % 对数均值
sigma = 0.8;         % 对数标准差

t = linspace(1, 1500, 1000);    % 0 时刻不取，避免除零

% 指数分布
R1   = exp(-lambda1 * t);
h1   = lambda1 * ones(size(t));
MTTF1 = 1/lambda1;

% 瑞利分布
R2 = exp(-(t.^2*lambda2)/2);
h2 = t * lambda2;
MTTF2 = 1 * sqrt(pi/2) /sqrt(lambda2);

% 威布尔分布
R3 = exp(-(t/ceita).^gama);
h3 = (gama/ceita) * (t/ceita).^(gama-1);
MTTF3 = ceita * gamma(1 + 1/gama);

% 对数正态分布
R4 = 1 - logncdf(t, mu, sigma);
h4 = pdf('logn', t, mu, sigma) ./ R4;
MTTF4 = exp(mu + sigma^2/2);

% 绘图
figure('Name','可靠度函数','Position',[100 100 900 400]);
plot(t, R1, 'r', 'LineWidth',1.5); hold on;
plot(t, R2, 'g');
plot(t, R3, 'b');
plot(t, R4,'k');
xlabel('Time t'); ylabel('可靠度函数 R(t)');
legend('指数分布','瑞利分布','威布尔分布','对数正态分布');
grid on;

figure('Name','失效率函数','Position',[100 250 900 400]);
plot(t, h1, 'r', 'LineWidth',1.5); hold on;
plot(t, h2, 'g');
plot(t, h3, 'b');
plot(t, h4,'k');
xlabel('Time t'); ylabel('失效率函数 h(t)');
legend('指数分布','瑞利分布','威布尔分布','对数正态分布');
grid on;

% 控制台输出 MTTF 
fprintf('=== MTTF (Mean Time To Failure) ===\n');
fprintf('指数: %.1f\n', MTTF1);
fprintf('瑞利:    %.1f\n', MTTF2);
fprintf('威布尔:     %.1f\n', MTTF3);
fprintf('对数:   %.1f\n', MTTF4);

%% 第二问：随机样本生成 + 直方图 + 参数点估计
clearvars -except lambda1 lambda2 gama ceita mu sigma   % 保留第一问给的参数
N = 500;        % 样本量
rng default     

%指数分布
T1  = exprnd(1/lambda1, [N,1]);           % 均值为1/lambda
figure; histogram(T1,20); hold on
xlabel('t'); legend('指数分布样本直方图');
lambda_1 = 1/mean(T1);
fprintf('指数分布   λ 估计 = %.6f\n',lambda_1);

%瑞利分布
T2 = raylrnd(1/sqrt(lambda2), [N,1]);    
figure; histogram(T2,20); hold on
xlabel('t');  legend('瑞利分布样本直方图');
b_hat = sqrt(2/pi)*mean(T2);
lambda2 = 1/b_hat^2;
fprintf('瑞利分布  sigma_R 估计 = %.6f\n',lambda2);

%威布尔分布
T3 = wblrnd(ceita, gama, [N,1]);     
figure; histogram(T3,20); hold on
xlabel('t');  legend('威布尔分布样本直方图');
[ceita,gama] = wblfit(T3);
fprintf('威布尔分布 尺度 估计 = %.6f   形状 估计 = %.6f\n',gama,ceita);

%对数正态分布
T4 = lognrnd(mu, sigma, [N,1]);
figure; histogram(T4,20); hold on
xlabel('t'); legend('对数正态分布样本直方图');
mu    = mean(log(T4));
sigma = std(log(T4));

fprintf('对数正态   μ 估计 = %.6f   σ 估计 = %.6f\n',mu,sigma);
