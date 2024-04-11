
% min{Y, G, H}  tr(Y'*P*Y*D^(-1)) + \lambda ||Y||_F^2 + \beta ||Y- G * H'||_F^2
% s.t. Y* 1 = 1, Y_ij >= 0, , D_jj = \sum_i{Y_ij}, G*1 = 1, G>=0, H'*H=I

clear
addpath('.\funs');
addpath('.\datasets');
dataname = 'JAFFE';
disp("------------【 " + dataname + " 】---------------");
load([dataname '.mat']);
try
    gt = double(Y);
    X = double(X);
catch
    gt = double(gt);
end
nC = length(unique(gt)); 
[nN, ~] = size(X); 

%% ==================== preprocessing ===================
X = data_process(X, "max-min");
P = mydistance(X, X, "knn-L2", fix(nN/nC));

%% ===================== parameters ======================

beta = 0.5;
lambda = 20;

G = AFCAGC_knn(nN, nC, lambda, beta, P);

time = toc;
[~, label] = max(G');
result = ClusteringMeasure(gt, label);
fprintf('$$ %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n',result(1:7));


