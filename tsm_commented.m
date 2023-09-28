function Y = tsm_commented(X,d,tau,opts)
% tsm: A function to perform some form of dimensionality reduction or transformation
% using thresholded similarity matrices and alternating minimization.
% 
% Input:
%   - X: p x n matrix, where p is dimensionality and n is the number of examples (input data)
%   - d: The desired dimensionality of output Y
%   - tau: Cosine threshold to create augmented inputs and compute similarity matrix
%   - opts: Structure containing various options:
%       maxIter: Maximum number of iterations for alternating minimization
%       tol: Tolerance for absolute improvement in RMSE cost to check convergence
%       momentum: Momentum for the update in alternating minimization
%       printout: Number representing how often to print the cost function
%
% Output:
%   - Y: d x n matrix representing the transformed, reduced-dimensionality data.

% Get the number of examples in the original input data
nx = size(X,2);

% Augment the input data based on the cosine threshold (tau)
X = augment_inputs(X,tau);

% Update the number of examples to include augmented examples
n = size(X,2);

% Compute the thresholded similarity matrix
S = compute_similarity(X,tau);
idxS = find(S); % Find the indices of non-zero elements in S
nonzerosS = single(nonzeros(S)); % Extract the non-zero elements and convert them to single precision

% Setup options for eigensolver
eig_opts.issym = 1; % The matrix is symmetric
eig_opts.tol = 1e-8; % Tolerance
eig_opts.fail = 'keep'; % If failure occurs, keep going
eig_opts.maxit = 500; % Maximum number of iterations

% Initialization
XXt = X*X'; % Compute the Gram matrix of X
Xfun = @(v) XXt*v; % Define an inline function to perform matrix-vector multiplication with XXt
[U,~] = eigs(Xfun, size(X,1), d, 'largestabs', eig_opts); % Perform eigendecomposition to get the initial Y
Y = U'*X; % Initialize Y
v = vecnorm(Y); % Compute the norm of each column in Y
Theta = (Y'*Y) - tau*(v'*v); % Initialize Theta
Zm = single(zeros(n)); % Initialize the momentum as zeros
maxSumZ = sum(sum(X,2).^2) - tau*sum(vecnorm(X))^2; % Maximum allowable sum of elements in Z

% Alternating Minimization
costP = Inf; % Initialize the previous cost to infinity
tStart = tic; % Start a timer
for iterate = 0:opts.maxIter
    % Update Z using Theta and non-zero elements of S
    Zn = min(0, Theta); % Initialize Zn with the minimum between 0 and Theta
    Zn(idxS) = nonzerosS; % Assign the non-zero elements of S to corresponding positions in Zn
    sumZ = sum(Zn,'all'); % Compute the sum of all elements in Zn
    % Ensure that the sum of elements in Zn does not exceed maxSumZ
    
%     This is some sort of clipping technique used here
    if (sumZ > maxSumZ)
        Zn = Zn + (maxSumZ - sumZ) / (numel(Zn) - nnz(S));
        Zn(idxS) = nonzerosS;
    end
    
    % Apply momentum if required
    if (iterate < 2 || opts.momentum == 0)
        Z = Zn;
    else
        Zp = Z; % Store the previous Z
        Z = Zn + opts.momentum * Zm; % Update Z with momentum
        Zm = Z - Zp; % Update the momentum
    end
    % Update Theta
    Zfun = @(v) Z*v; % Define an inline function to perform matrix-vector multiplication with Z
    [V,D] = eigs(Zfun, n, d, 'largestabs', eig_opts); % Perform eigendecomposition to update Theta
    Theta = V*D*V'; % Update Theta
    % Compute cost as RMSE between Z and Theta
    cost = sqrt(mean((Z - Theta).^2, 'all'));
    % Print the cost at specified intervals
    if (mod(iterate, opts.printout) == 0)
        fprintf(1, '  iterate: %03d  cost: %.5f   tsec: %f\n', iterate, cost, toc(tStart));
        tStart = tic; % Restart the timer
    end
    % Check for convergence
    if (opts.momentum == 0 && (costP - cost) < costP * opts.tol)
        break;
    else
        costP = cost; % Update the previous cost
    end
end

% Compute the final embedding Y using the Gram matrix G
u = sqrt(max(0, diag(Theta))); % Compute the square root of the diagonal elements of Theta
G = Theta + (tau / (1 - tau)) * (u * u'); % Compute the Gram matrix G
Gfun = @(v) G * v; % Define an inline function to perform matrix-vector multiplication with G
[vG,eG] = eigs(Gfun, n, d, 'largestreal', eig_opts); % Perform eigendecomposition to get the final Y
Y = sqrt(max(0, eG)) * vG'; % Update Y using the obtained eigenvalues and eigenvectors
Y = single(Y(:, 1:nx)); % Return Y with single precision and original number of examples
return;
end


% % % % % % % % % % % % % % % % % % % % % % % % % % 


function S = compute_similarity(X,tau)
% compute_similarity: Computes a sparse similarity matrix based on a threshold (tau).
% 
% Input:
%   - X: The input data matrix
%   - tau: The threshold for computing the similarity matrix
% 
% Output:
%   - S: The computed sparse similarity matrix

% Compute the norm of each column in X
v = vecnorm(X);

% Compute the sparse similarity matrix S
S = sparse(double(max(0,X'*X - tau*(v'*v))));
return;
end


% % % % % % % % % % % % % % % % % % % % % % % % % % % 

function X = augment_inputs(X,tau)
% X: The input matrix where each column is a data vector
% tau: The cosine similarity threshold

% n: Number of columns in input matrix X
n = size(X,2);

% v: Vector containing the norms of the columns of X
v = vecnorm(X);

% cosines: Matrix containing cosine similarities between columns of X
cosines = (X'*X)./(v'*v);
cosines(1:n+1:end) = 0; % Remove self-similarities by setting diagonal to 0

% nnCos: The maximum cosine similarity for each column
% nnIdx: The corresponding column index for each maximum cosine similarity
[nnCos,nnIdx] = max(cosines);

% Initialize the matrix for augmented examples
Xa = [];

% Iterate over each column in X
for i=1:n
  % Skip if maximum cosine similarity is above the threshold
  if (nnCos(i)>=tau)
    continue;
  end

  % Get the column index with which the ith column has maximum cosine similarity
  j = nnIdx(i);

  % u1, u2: Normalized versions of the columns i and j of X
  u1 = X(:,i)/v(i);
  u2 = X(:,j)/v(j);

  % m: Number of interpolation steps needed between u1 and u2
  m = ceil(acos(nnCos(i))/acos(tau));

  % Interpolate and augment
  for k=1:m-1
    % Interpolate between the normalized vectors u1 and u2
    xk = (k/m)*u1 + (1-k/m)*u2;
    
    % Scale the interpolated vector xk by the interpolated norm
    mk = (k/m)*v(i) + (1-k/m)*v(j);
    xk = xk*(mk/norm(xk));
    
    % Append the new interpolated column vector to Xa
    Xa = [Xa xk];
  end
end

% Combine the original and augmented matrices
X = [X Xa];
end

