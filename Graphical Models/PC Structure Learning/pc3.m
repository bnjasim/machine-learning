% PC3 Algorithm
function A = pc3(D)

    % number of variables
    n = size(D, 2);

    n_tests = 0; % number of CItests conducted

    % Find the Moral Graph
    MG = zeros(n, n);
    % For all pairs of variables do the required check to add an edge
    for X = 1:n-1
        for Y = X+1:n
            set = 1:n;
            set([X,Y]) = []; % remove X & Y from set
            set = transpose(set);
            c = CItest(D, X, Y, set);
            n_tests = n_tests + 1; 

            if (c == 0)
                % Add an edge in the moral graph
                MG(X, Y) = 1;
                MG(Y, X) = 1;
            end
        end
    end


    % Initialize Adjacency Matrix
    % Start with all nodes connected initially
    A = ones(n, n) - eye(n); % no self loops

    % Iterate over all pairs of variables
    for X = 1:n-1
        for Y = X+1:n
            % Remove the edge b/w X and Y if CItest returns 1 for  
            % any subset of the remaining variables.
            set = 1:n;

            % Finding subsets of all remaining variables which are 
            % neighbours to X & Y in the moral graph

            nX = set(MG(X,:) == 1); % neighbours of X
            nX = nX(nX ~= Y); % Remove Y from neighbours of X
            nY = set(MG(Y,:) == 1); % neighbours of Y
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