%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SETUP

clc;
clearvars;
format compact;
subsample_ratio = 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DATA & SUBSAMPLE

tStart = tic;
fprintf(1,'\nLoading MNIST digits and precomputed kNN\n');
load('digits_70k_64nn.mat','images','knnIdx');
X = single(images)/single(intmax('uint8'));
subIdx = 1:subsample_ratio:size(X,2);
Xs = X(:,subIdx);
clear images;
toc(tStart);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SIMILARITY MATRIX

tStart = tic;
fprintf(1,'\nComputing thresholded similarity matrix on subsample ...\n');
tau = 0.75;
v = vecnorm(Xs);
S = sparse(double(max(0,(Xs'*Xs)-tau*(v'*v))));
numSim = full(sum(S>0));
fprintf(1,'Number of examples: %d\n',size(S,1))
fprintf(1,'Cosine threshold: %f\n',tau);
fprintf(1,'Number of disconnected examples: %d\n',nnz(numSim==1));
fprintf(1,'Number of singly connected examples: %d\n',nnz(numSim==2));
fprintf(1,'Med/Mean/Max neighbors: %d %.1f %d\n',...
  median(numSim)-1,mean(numSim)-1,max(numSim)-1);
fprintf(1,'Sparsity: %f\n',1-nnz(S)/numel(S));
toc(tStart);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TSM

tStart = tic;
d = 16;
fprintf(1,'\nComputing TSM for d=%d ...\n',d);
opts.maxIter = 250;
opts.momentum = 0.9;
opts.printout = 10;
opts.tol = 0;
Ys = tsm(Xs,d,tau,opts);
[angErr1,jaccard1] = compare_XY(Xs,Ys,tau);
fprintf(1,'mean angular deviation on subsample: %f\n',angErr1);
fprintf(1,'jaccard index on subsample: %f\n',jaccard1);
toc(tStart);

% LLX
tStart = tic;
k = 4*d;
fprintf(1,'\nComputing LLX for k=%d ...\n',k);
Y = llx(X,knnIdx(1:k,:),Ys,subIdx);
[angErr2,jaccard2] = compare_XY(X,Y,tau);
fprintf(1,'mean angular deviation on whole: %f\n',angErr2);
fprintf(1,'jaccard index on whole: %f\n',jaccard2);
toc(tStart);

% SVD
tStart = tic;
fprintf(1,'\nComputing SVD for d=%d...\n',d);
[U,~,~] = svd(X,'econ');
[Uf,~,~] = svd(Xs,'econ');
Y = U(:,1:d)'*X;
Ys = Uf(:,1:d)'*Xs;
[angErr1,jaccard1] = compare_XY(Xs,Ys,tau);
fprintf(1,'mean angular deviation on subsample: %f\n',angErr1);
fprintf(1,'jaccard index on subsample: %f\n',jaccard1);
[angErr2,jaccard2] = compare_XY(X,Y,tau);
fprintf(1,'mean angular deviation on whole: %f\n',angErr2);
fprintf(1,'jaccard index on whole: %f\n',jaccard2);
toc(tStart);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

