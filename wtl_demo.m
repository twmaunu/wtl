%% G(2,1) energy landscape for a random dataset - note that changing 
%   the random number generator (or not seeding) may result in datasets 
%   with local minima

rng(42,'twister')

nout = 90;
nin = 10;

v = [0;1];
inliers = (v*randn(1,nin))';
outliers = randn(nout,2)/2;

points = [inliers;outliers];

angles=0:0.01:pi;
cost=zeros(1,length(angles));
for j=1:length(angles)
    vhat = [cos(angles(j));sin(angles(j))];
    diffs = points - (points * vhat) * vhat';
    cost(j) = sum(sum(diffs.^2,2).^.5);
end

figure
plot(inliers(:,1),inliers(:,2),'x','markersize',10,'linewidth',2)
hold on
plot(outliers(:,1),outliers(:,2),'o','markersize',10,'linewidth',2)
grid on
legend('Inliers','Outliers')
set(gca,'fontname','Times')
set(gca,'fontsize',24)

figure
plot(angles/pi,cost,'linewidth',2)
xlim([angles(1)/pi,angles(end)/pi+.001])
xlabel('\theta (multiple of \pi)')
ylabel('Energy')
set(gca,'fontname','Times')
set(gca,'fontsize',24)


figure
plot(angles/pi,cost,'linewidth',2)
xlim([angles(1)/pi,angles(end)/pi+.001])
xlabel('\theta (multiple of \pi)')
ylabel('Energy')
set(gca,'fontname','Times')
set(gca,'fontsize',24)

%% Plot energy landscape for G(2,1) with one point

rng(42,'twister')

nin = 1;

points=[0,1];
angles=0:0.01:pi;
cost=zeros(1,length(angles));
for j=1:length(angles)
    vhat = [cos(angles(j));sin(angles(j))];    
    diffs = points - (points * vhat) * vhat';
    cost(j) = sum(sum(diffs.^2,2).^.5);
end
plot(angles/pi,cost)
xlim([angles(1)/pi,angles(end)/pi])
xlabel('Angle (multiple of \pi)')
ylabel('Cost')
set(gca,'fontname','Times')
set(gca,'fontsize',24)

%% G(3,1) energy landscape

rng(42,'twister')
vstar = [0,0,1]';
U = [1,0,0;0,1,0]';

theta = 0:.005:(2*pi+.05);
phi = 0:.005:(pi/2+.05);

inliers = vstar * randn(1,20);
% inliers = inliers + 1e-3*randn(size(inliers,1),size(inliers,2));
outliers = randn(3,100) / sqrt(3)*2.0;
points = [inliers,outliers];

points = points + .001*randn(size(points,1),size(points,2));

x = zeros(length(theta),length(phi));
y = zeros(length(theta),length(phi));
cost = zeros(length(theta),length(phi));
for i=1:length(theta)
    for j=1:length(phi)
        v = [cos(theta(i))*sin(phi(j)),sin(theta(i))*sin(phi(j)),cos(phi(j))]';
        x(i,j) = cos(theta(i))*sin(phi(j));
        y(i,j) = sin(theta(i))*sin(phi(j));
        tmp = points - v * (v' * points);
        tmp = sum(tmp .* tmp).^.5;
        cost(i,j) = sum(tmp);

    end
end

figure
plot3(inliers(1,:),inliers(2,:),inliers(3,:),'x','markersize',10,'linewidth',2)
hold on
plot3(outliers(1,:),outliers(2,:),outliers(3,:),'o','markersize',10,'linewidth',2)
grid on
legend('Inliers','Outliers','Location','Northeast')
set(gca,'fontname','Times')
set(gca,'fontsize',24)

figure
phi2 = repmat(phi,length(theta),1);
theta2 = repmat(theta',1,length(phi));
surf(x,y,cost,'EdgeColor','none')
axis off
colorbar
set(gca,'fontsize',24)


%% Simulation of stability condition

rng(42,'twister')

% generate data
d = 10;
D = 200;
Nin = 200;
Nout = 200;
U = orth(randn(D,d));
inliers = randn(Nin,d) *  U'/d;
noise = 0;
Q = eye(D) - U * U';
innorms = sum(inliers.^2,2).^.5;
inliersn = inliers + noise^.5 * repmat(innorms,1,D) .* randn(size(inliers,1),size(inliers,2)) * Q / (D-d);
bas = eye(D);
outCov = bas * diag(rand(1,D)) * bas'/D;
outliers = randn(Nout,D) * outCov^.5 ;
X = [inliersn;outliers];

% set parameter gamma and eta
gamma = pi/4;    
eta = 2*atan(noise^.5);


dist = eta:.02:gamma;   % principal angles with L^*
reps = 20;              % simulate 20 subspaces per distance
QU = eye(D) - U*U';

% calculate value of inlier term
tmpin = inliersn;
tmpin = tmpin * U * U';
tmpin = tmpin ./ repmat(sum(tmpin.^2,2).^.25,1,D);
[u,s,v] = svd(tmpin);
inval = s(d,d)^2*cos(gamma+eta)*cos(eta);

% Calculate outlier value for random V with maximum principal angle dist(i)
outvals = zeros(length(dist),reps);  
for i=1:length(dist)
    for k=1:reps
        dir = orth(QU*randn(D,d));
        angles = rand(1,d);
        angles = angles/max(angles);
        angles = angles*dist(i);
        V = U*diag(cos(angles)) + dir*diag(sin(angles));
        
        
        outliers_dot_V = outliers * V;
        dists = sum((outliers' - V * (V' * outliers')).^2).^.5;
        dists = dists';
        dists(dists==0) = inf;
        dists = repmat(dists,1,d);
        
        scale = outliers_dot_V ./ dists;
        gradFVoutliers = zeros(d,D);
        for j=1:d
            gradFVoutliers(j,:) = sum(outliers .* repmat(scale(:,j),1,D));
        end
        pgradFVoutliers = gradFVoutliers' - V * (V' * gradFVoutliers');
        
        [u,s,v] = svd(pgradFVoutliers);
        outvals(i,k) = s(1,1);
        
    end
end

diffvals = inval-outvals;
imagesc(1:reps,dist/pi,diffvals)
h=colorbar;
set(h,'fontsize',20);
xlabel('Index of randomly generated subspace')
ylabel('\Theta_1 with L^* (multiple of \pi)')
set(gca,'fontname','Times')
set(gca,'fontsize',22)

%% Example landscape with local minima

phis = 0.1*pi:.25*pi:2*pi;
inliers = .5*[0,0,0,0;1,-1,.5,-.5]';
outliers = [cos(phis);sin(phis)]';
points = [inliers;outliers];

angles=0:0.01:pi;
cost=zeros(1,length(angles));
for j=1:length(angles)
    cost(j)=0;
    vhat = [cos(angles(j));sin(angles(j))];
    diffs = points - (points * vhat) * vhat';
    cost(j) = sum(sum(diffs.^2,2).^.5);
end

figure
plot(inliers(:,1),inliers(:,2),'x','markersize',15,'linewidth',3)
hold on
plot(outliers(:,1),outliers(:,2),'o','markersize',15,'linewidth',3)
grid on
legend('Inliers','Outliers')
set(gca,'fontname','Times')
set(gca,'fontsize',24)

figure
plot(angles/pi,cost,'linewidth',5)
xlim([angles(1)/pi,angles(end)/pi+.001])
xlabel('\theta (multiple of \pi)')
ylabel('Energy')
set(gca,'fontname','Times')
set(gca,'fontsize',24)

%% Plot haystack and noisy haystack

rng(42,'twister')


% haystack model
nout = 50;
nin = 50;
points=rand(1,nout)*2*pi;
v = [0;1];
inliers = (v*randn(1,nin))';
outliers = randn(nout,2)/sqrt(2);

figure
plot(inliers(:,1),inliers(:,2),'x','markersize',10,'linewidth',2)
hold on
plot(outliers(:,1),outliers(:,2),'o','markersize',10,'linewidth',2)
grid on
% legend('Inliers','Outliers')
set(gca,'fontname','Times')
set(gca,'fontsize',24)
xlim([-2,2])
ylim([-2,2])


% noisy haystack model
sigmanoise = (1e-1)^2;
innorms = sum(inliers.^2,2).^.5;
inliersn = inliers + sqrt(sigmanoise)*repmat(innorms,1,2).^.5 .* ([1;0]*randn(1,nin))';
pts = [inliersn;outliers];
angles = acos(sum((pts * v*v').^2,2).^.5 ./ sum((pts).^2,2).^.5);
slope = 1/(30*sigmanoise);
athresh = 1/slope;

figure
plot(inliersn(:,1),inliersn(:,2),'x','markersize',10,'linewidth',2)
hold on
plot(outliers(:,1),outliers(:,2),'o','markersize',10,'linewidth',2)
grid on
legend('Inliers','Outliers')
set(gca,'fontname','Times')
set(gca,'fontsize',24)
xlim([-2,2])
ylim([-2,2])

plot([-1 1],[-slope slope],'k')
slope = -slope;
plot([-1 1],[-slope slope],'k')


