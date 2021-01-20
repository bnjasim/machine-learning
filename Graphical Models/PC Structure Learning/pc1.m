% PC1 Algorithm
function A = pc1(D)

    % number of variables
    n = size(D, 2);

    % Initialize Adjacency Matrix
    % Assume there are no edges initially
    A = zeros(n, n);

    n_tests = 0; % number of CItests conducted

    % Iterate over all pairs of variables
    for X = 1:n-1
        for Y = X+1:n
            % Add an edge b/w X and Y if CItest returns 0 for  
            % all subsets of the remaining variables.
            % If CItest returns 1 for any subset, then add no edge
            % and we can move to the next pair of variables

            % Finding subsets of all remaining variables
            set = 1:n;
            set([X,Y]) = []; % remove X & Y from set
            S = subsets(set); % find all subsets of set
            
            % for each subset check CI
            flag = 0;
            for i = 1:numel(S)
                Z = transpose(S{i});
                c = CItest(D, X, Y, Z); % check conditional independence
                n_tests = n_tests + 1; 
                if (c == 1)
                    flag = 1;
                    break;
                end
            end

            % flag will be 0 if CItest returns 0 for all subsets
            if (flag == 0)
                % Add an edge between X & Y in the Adjacency Matrix
                A(X, Y) = 1; 
                A(Y, X) = 1;
            end
        end
    end
    
    disp(n_tests);
end