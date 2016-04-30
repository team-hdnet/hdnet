function [J, theta] = hdnet_fit_ising(X)
    % hdnet_fit_ising: Fit Ising model to binary data X using minimum
    % probability flow (MPF). Needs Python package hdnet
    % (https://github.com/team-hdnet/hdnet).
    %
    % INPUT
    % X: n x m matrix of binary (integer 0, 1) observations
    % n neurons, m observations
    %
    % OUTPUT
    % J: n x n matrix of Ising coupling strengths
    % theta: n x 1 vector of Ising biases
    
    N = size(X, 1);
    hdnet = py.importlib.import_module('hdnet_matlab');
    ret = cell(hdnet.fit_ising_matlab(N, X(:)'));
    J = reshape(cellfun(@double, cell(ret{1})), [N N]);
    theta = reshape(cellfun(@double, cell(ret{2})), [N 1]);    
end

