clear; clc; close all;
% 基本参数
lambda = [1e-4, 3e-4, 3e-4, 3e-4, 2e-4, 2e-4]; 
n = 6;          % 组件数量
s = 50000;     % 模拟次数
t_max = 10000;       % 最大时间
t_step = 50;         % 时间步长
t = 0:t_step:t_max;  % 时间序列
n_t = length(t);     % 时间点数量
% 初始化
R_est = zeros(1, n_t);  % 可靠度估计值
CI_low = zeros(1, n_t); % 置信区间下限
CI_up = zeros(1, n_t);  % 置信区间上限
% 生成各组件寿命
mttf = -log(rand(s, n)) ./ lambda;
f = @(x) x(1) & (x(2) | x(3) | x(4)) & (x(5) | x(6));
% 蒙特卡洛模拟
for k = 1:n_t
    t_now = t(k);
    % 组件寿命>当前时间为正常(1)，否则故障(0)
    state = mttf > t_now;
    % 计算系统状态
    sys_state = false(s, 1);
    for i = 1:s
        sys_state(i) = f(state(i, :));
    end
    % 可靠度估计
    R_est(k) = mean(sys_state);
    % 95%置信区间
    z = norminv(0.975);  % 1.96
    se = sqrt(R_est(k)*(1-R_est(k))/s); 
    me = z * se;       
    CI_low(k) = max(0, R_est(k)-me);
    CI_up(k) = min(1, R_est(k)+me);
   end

% 绘图
figure('Position', [100, 100, 900, 600]);
plot(t, R_est, 'b-', 'LineWidth', 2);
hold on;
plot(t, CI_low, 'r--', 'LineWidth', 1);
plot(t, CI_up, 'r--', 'LineWidth', 1);
xlabel('时间 t (小时)');
ylabel('系统可靠度 R_s(t)');
title('蒙特卡洛模拟：系统可靠度函数及95%置信区间');
legend('可靠度估计值', '95%置信区间下限', '95%置信区间上限', 'Location', 'northeast');
grid on;
xlim([0, t_max]);
ylim([0, 1]);
text(0.02*t_max, 0.15, sprintf('模拟次数: %d\n时间步长: %d 小时', s, t_step), ...
     'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');
hold off;