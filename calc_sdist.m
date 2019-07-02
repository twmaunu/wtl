function [ dist ] = calc_sdist( S1 , S2 )
%CALC_SDIST Summary of this function goes here
%   Calculate distance between subspaces using sqrt of sum of squared 
%       principal angles
%   S1,S2 orthogonal column bases for subspaces
% T Maunu 2017

    A = S1'*S2;
    [u,s,v] = svd(A);
    s = diag(s);
    for i=1:size(s,1)
        s(i,1) = acos(s(i,1));
    end
    dist = s'*s;
    dist = dist^.5;
end

