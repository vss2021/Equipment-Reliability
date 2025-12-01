clear; clc; close all;

%%
% 极限状态函数
g = @(x1, x2) (x1 - 1).^3 + (x2 - 2).^2 + x1 .* sin(2*pi*x2) .* cos(2*pi*x1) + 10;
g_fun = @(X) g(X(:,1), X(:,2));

% 随机变量参数（标准正态）
mu = [0; 0];
sigma = [1 0; 0 1];
n = 2;

% 计时变量
t1 = 0; t2 = 0; t3 = 0; t4 = 0; t5 = 0;

%% 1. 一次二阶矩法
tic;
x0 = mu;
g0 = g(x0(1), x0(2));

% 数值求导
eps = 1e-6;
dg1 = (g(x0(1)+eps, x0(2)) - g(x0(1)-eps, x0(2)))/(2*eps);
dg2 = (g(x0(1), x0(2)+eps) - g(x0(1), x0(2)-eps))/(2*eps);
dg = [dg1; dg2];

beta = g0 / sqrt(dg'*sigma*dg);
pf1 = normcdf(-beta);
t1 = toc;

%% 2. 蒙特卡洛法
tic;
N = 1e6;
X = mvnrnd(mu', sigma, N);
g_val = g_fun(X);
pf2 = sum(g_val <= 0)/N;
t2 = toc;

%% 3. 重要抽样法
tic;
% 找抽样中心
x1 = -3:0.1:3; x2 = -3:0.1:3;
[X1,X2] = meshgrid(x1,x2);
G = g(X1,X2);
idx = find(abs(G)<1);

if ~isempty(idx)
    Xc = [X1(idx), X2(idx)];
    pdf = mvnpdf(Xc, mu', sigma);
    [~, id] = max(pdf);
    mu_is = Xc(id,:)';
else
    mu_is = mu;
end

% 抽样计算
Nis = 1e5;
Xis = mvnrnd(mu_is', sigma, Nis);
gis = g_fun(Xis);
w = mvnpdf(Xis, mu', sigma)./mvnpdf(Xis, mu_is', sigma);
pf3 = mean(w.*(gis<=0));
t3 = toc;

%% 4. 子集模拟法
tic;
Nss = 1e5;  % 减小样本量，加快速度
p0 = 0.8;   
maxl = 50;
xmin = -5; xmax = 5;
step = 0.1;

% 初始样本
Xp = mvnrnd(mu', sigma, Nss);
Xp(Xp<xmin) = xmin; Xp(Xp>xmax) = xmax;
gp = g_fun(Xp);
pf4 = 1;
level = 0;

while true
    level = level + 1;
    gs = sort(gp);
    k = max(1, floor(p0*Nss));
    gk = gs(k);
    
    pf4 = pf4 * (k/Nss);
    
    % 终止条件
    if gk <= 0
        pf4 = pf4 * sum(gp<=0)/Nss;
        break;
    end
    
    % 条件样本
    Xc = Xp(gp<=gk, :);
    m = size(Xc,1);
    if m == 0
        pf4 = 0;
        break;
    end
    
    % MCMC抽样
    Xn = zeros(Nss, n);
    pos = 1;
    nchain = ceil(Nss/m);
    
    for j=1:m
        xs = Xc(j,:);
        for k=1:nchain
            if pos>Nss, break; end
            xc = xs;
            acc = false;
            while ~acc
                xp = xc + step*randn(1,n);
                xp(xp<xmin)=xmin; xp(xp>xmax)=xmax;
                gp = g(xp(1), xp(2));
                if gp <= gk
                    acc = true;
                    xc = xp;
                end
            end
            Xn(pos,:) = xc;
            pos = pos + 1;
        end
    end
    
    Xp = Xn;
    gp = g_fun(Xp);
    
    if level>=maxl
        warning('达到最大层数');
        pf4 = pf4 * sum(gp<=0)/Nss;
        break;
    end
end
t4 = toc;

%% 5. AK-MCS法
tic;
% 初始样本
N0 = 50;
X0 = lhsdesign(N0, n)*10 -5;
Y0 = g_fun(X0);

% 构建Kriging模型
model = fitrgp(X0, Y0, 'KernelFunction', 'squaredexponential');

% 自适应加点
Nmax = 100;
for i=1:Nmax-N0
    Xc = lhsdesign(1000, n)*10 -5;
    [Yp, Yv] = predict(model, Xc);
    s = sqrt(Yv);
    U = abs(Yp)./s;
    [~, id] = min(U);
    Xn = Xc(id,:);
    Yn = g_fun(Xn);
    X0 = [X0; Xn];
    Y0 = [Y0; Yn];
    model = fitrgp(X0, Y0, 'KernelFunction', 'squaredexponential');
end

% 蒙特卡洛
Nak = 1e6;
Xak = mvnrnd(mu', sigma, Nak);
Yak = predict(model, Xak);
pf5 = sum(Yak<=0)/Nak;
t5 = toc;

%% 结果输出
fprintf('===============================\n');
fprintf('方法          失效概率        时间(s)\n');
fprintf('===============================\n');
fprintf('一次二阶矩法   %.6e      %.4f\n', pf1, t1);
fprintf('蒙特卡洛       %.6e      %.4f\n', pf2, t2);
fprintf('重要抽样法     %.6e      %.4f\n', pf3, t3);
fprintf('子集模拟法     %.6e      %.4f\n', pf4, t4);
fprintf('AK-MCS法       %.6e      %.4f\n', pf5, t5);
fprintf('===============================\n');