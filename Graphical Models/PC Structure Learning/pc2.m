% PC2 Algorithm
function A = pc2(D)

    % number of variables
    n = size(D, 2);

    % Initialize Adjacency Matrix
    % Start with all nodes connected initially
    A = ones(n, n) - eye(n); % no self loops

    n_tests = 0; % number of CItests conducted

    % Iterate over all pairs of variables
    for X = 1:n-1
        for Y = X+1:n
            % Remove the edge b/w X and Y if CItest returns 1 for  
            % any subset of the remaining variables.

            % Finding subsets of all remaining variables
            set = 1:n;
            % Finding subsets of all remaining variables which are 
            % neighbours to X & Y in the current graph

            nX = set(A(X,:) == 1); % neighbours of X
            nX = nX(nX ~= Y); % Remove Y from neighbours of X
            nY = set(A(Y,:) == 1); % neighbours of Y
            nY = nY(nY ~= X); % Remove X from neighbours of Y
            nXY = union(nX, nY);
            S = subsets(nXY); % find all subsets of the neighbours
            
            % for each subset check CI
            for i = 1:numel(S)
                Z = transpose(S{i});
                c = CItest(D, X, Y, Z); % check conditional independence
                n_tests = n_tests + 1; 
                if (c == 1)
                    % Remove the edge between X & Y
                    A(X, Y) = 0;
                    A(Y, X) = 0;
                    break;
                end
            end

        end
    end

    disp(n_tests);
end