%% Example comparison of robust subspace to PCA
% Gaussian inliers/outliers, symmetric
%
% (line search convergence is much faster, use opt = 1)

d = 5;
D = 100;
Nin = 100;
Nout = 100;

U = orth(randn(D,d));
inliers = randn(Nin,d) *  U' / sqrt(d);
outliers = randn(Nout,D) / sqrt(D);
X = [inliers;outliers];

opt = 0;
[vhat,convdist] = ggd(X,.1/(Nin+Nout),10000,d,opt,U);
calc_sdist(vhat,U)

[u,s,v] = randpca(X,d);
calc_sdist(v,U)



%% Convergence plot from paper
% Gaussian inliers/outliers, symmetric

rng(123,'twister')

d = 5;
D = 100;
Nin = 200;
Nout = 200;

U = orth(randn(D,d));
inliers = randn(Nin,d) *  U';
outliers = randn(Nout,D) ;
X = [inliers;outliers];

[vhat,convdist_sqrtk] = ggd(X,1/(Nin+Nout),1000,d,0,U);
calc_sdist(vhat,U)

[vhat,convdist_linesearch] = ggd(X,1/(Nin+Nout),100,d,1,U);
calc_sdist(vhat,U)
convdist_linesearch(convdist_linesearch==0) = 2e-8;   % for log scale

[vhat,convdist_shrinking] = ggd(X,1,1000,d,2,U);
calc_sdist(vhat,U)
convdist_shrinking(convdist_shrinking==0) = 2e-8;   % for log scale


figure
plot(log10(convdist_sqrtk),'-b','markersize',5,'linewidth',2)
hold on
plot(log10(convdist_linesearch),'-r','markersize',5,'linewidth',2)
plot(log10(convdist_shrinking),'-g','markersize',5,'linewidth',2)

xlabel('Iteration')
ylabel('log \Theta_1 with L^*')
set(gca,'fontname','Times')
set(gca,'fontsize',24)
legend('s=1/N','Linesearch','Shrinking s')
ylim([-8,3])
grid on
