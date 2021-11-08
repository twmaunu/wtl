function [ Vk, conv ] = ggd( X, s, maxiter, d, opt, true_sub )
%GGD 
%
% X:        N times D data matrix
% s:        initial step size
% maxiter:  number of iteration
% d:        subspace dimension
% opt:      0 for s/sqrt(k) step size, 
%           1 for line search step size,
%           2 for shrinking s
% trueSub:  true subspace (for convergence plot)
%
%   Implementation of geodesic gradient descent from the paper "A Well
%   Tempered Landscape for Non-convex Robust Subspace Recovery" 
%        https://arxiv.org/abs/1706.03896
%
%   T Maunu 2017

if nargin < 6
    true_sub = orth(randn(size(X,2),d));
end
addpath('random pca')

[N,D] = size(X);

[u0,s0,v0] = svd(X);
Vk = v0(:, 1:d);

tol = 1e-10;
Vprev = Vk;
seq_dist = 1;
k = 1;
conv = [];

while k<maxiter && seq_dist > tol
    
    % calculate gradient
    X_dot_vk = X * Vk;
    dists = sum((X' - Vk * (Vk' * X')).^2).^.5;
    dists(dists < 1e-12 ) = inf;   % points on subspace contribute 0
    dists = repmat(dists',1,d);
    scale = X_dot_vk ./ dists;
    
    derFvk = (X' * scale)';
    
    % gradient is projected derivative
    gradFvk = derFvk' - Vk * (Vk' * derFvk');
    
    % SVD for geodesic calculation
    [U,Sigma,W] = svd(gradFvk,'econ');
    
    if opt == 1
        %line search
        step = s;
        cond = 1;
        currCost = cost(Vk,X);
        while cond
            Vkt = Vk*W*diag(cos(diag(Sigma*step))) + U*diag(sin(diag(Sigma*step)));
            if cost(Vkt,X) < currCost || step <= 1e-16
                Vk = Vkt;
                cond = 0;
            else
                step = step / 2;
            end           
        end      
    elseif opt == 2   
        % shrinking s
        step = s;
        Vkt = Vk*W*diag(cos(diag(Sigma*step))) + U*diag(sin(diag(Sigma*step)));
        Vk = Vkt;
        if mod(k,50)==0
            s = s / 10;
        end
    else
        % 1/sqrt(k)
        step = s/sqrt(k);
        Vkt = Vk*W*diag(cos(diag(Sigma*step))) + U*diag(sin(diag(Sigma*step)));
        Vk = Vkt; 
    end

    % calculate maximum principal angle between A and true subspace (for plots)
    A = true_sub'*Vk;
    [~,sv,~] = svd(A);
    sv = diag(sv);
    for l=1:size(sv,1)
        if sv(l,1)>=1
            sv(l,1) = 0;
        else
            sv(l,1) = acos(sv(l,1));
        end
    end
    conv = [conv max(sv)];  
    
    k = k + 1;
    seq_dist = calc_sdist(Vk, Vprev);
    Vprev = Vk;

end

end


function [out] = cost(v,X)
% calculate cost function from paper

out = sum(sum((X' - v * (v' * X')).^2).^.5);

end

