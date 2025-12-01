clear; clc; close all;

% 参数设定
gama = 5;     % 形状参数
theta = 20;   % 尺度参数
N = 40;       % 样本量
n_rep = 1000; % 模拟次数
s_sizes = [10, 30, 50, 100, 200]; % 不同样本量
% 生成威布尔分布样本
S = wblrnd(theta, gama, N, 1);
% 指数分布参数估计
lambda_exp = 1/mean(S); 
% 威布尔分布矩估计
mu = mean(S);                  % 样本均值
m2 = mean(S.^2);               % 样本二阶原点矩
Cv2 = (m2 / mu^2) - 1;         % 变异系数平方
fun_g = @(g) (gamma(1 + 2/g) / (gamma(1 + 1/g))^2) - 1 - Cv2;% 方程逻辑：[Gamma(1+2/k)/Gamma(1+1/k)^2] - 1 = Cv2
g_low = 0.5;  
g_high = 10;
tol = 1e-6;   % 迭代精度
while (g_high - g_low) > tol
    g_mid = (g_low + g_high) / 2;
    f_mid = fun_g(g_mid);
    if f_mid > 0
        g_low = g_mid;
    else
        g_high = g_mid;
    end
end
g_jv = g_mid;
theta_jv = mu / gamma(1 + 1/g_jv); 
% 最大似然估计
[shape_weib_mle, scale_weib_mle] = wblfit(S);
shape_weib_mle = shape_weib_mle(1);
scale_weib_mle = scale_weib_mle(1);
% 绘制密度函数对比
x = linspace(0, 2*theta, 200);
f_true = wblpdf(x, theta, gama);          % 真实分布
f_exp = exppdf(x, 1/lambda_exp);          % 指数分布估计
f_weib_mom = wblpdf(x, theta_jv, g_jv);   % 矩估计分布
f_weib_mle = wblpdf(x, scale_weib_mle, shape_weib_mle); % 最大似然分布

figure('Position', [100, 100, 800, 500]);
plot(x, f_true, 'k-', 'LineWidth', 2, 'DisplayName', '真实威布尔分布');
hold on;
plot(x, f_exp, 'r--', 'DisplayName', '指数分布估计');
plot(x, f_weib_mom, 'b-.', 'LineWidth', 1.5, 'DisplayName', '威布尔矩估计');
plot(x, f_weib_mle, 'g:', 'LineWidth', 2, 'DisplayName', '威布尔最大似然估计');
xlabel('寿命'); ylabel('概率密度');
legend('Location', 'best');
title(sprintf('寿命分布密度函数对比（N=%d）', N));
grid on;
hold off;

% 不同样本量下的变异系数分析
gama_dif = zeros(n_rep, length(s_sizes));
theta_dif = zeros(n_rep, length(s_sizes));

for i = 1:length(s_sizes)
    N_i = s_sizes(i);
    for j = 1:n_rep
        S_j = wblrnd(theta, gama, N_i, 1);
        [gama_j, theta_j] = wblfit(S_j);
        gama_dif(j,i) = gama_j(1);
        theta_dif(j,i) = theta_j(1);
    end
end

% 计算变异系数
cv_gama = std(gama_dif, 0, 1) ./ mean(gama_dif, 1);
cv_theta = std(theta_dif, 0, 1) ./ mean(theta_dif, 1);

figure('Position', [100, 200, 800, 500]);
plot(s_sizes, cv_gama, 'ro-', 'MarkerSize', 6, 'DisplayName', '形状参数变异系数');
hold on;
plot(s_sizes, cv_theta, 'bs-', 'MarkerSize', 6, 'DisplayName', '尺度参数变异系数');
xlabel('样本量N'); ylabel('变异系数（标准差/均值）');
legend('Location', 'best');
title('威布尔参数MLE估计的变异系数随样本量的变化');
grid on;
hold off;

fprintf('形状参数：%.4f，尺度参数：%.4f\n', gama, theta);
fprintf('威布尔矩估计 - 形状：%.4f，尺度：%.4f\n', g_jv, theta_jv);
fprintf('威布尔MLE - 形状：%.4f，尺度：%.4f\n', shape_weib_mle, scale_weib_mle);