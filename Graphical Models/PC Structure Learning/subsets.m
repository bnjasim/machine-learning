function S = subsets(set)
    % return all the subsets of a set 
    R = cell(1, numel(set) + 1);
    R{1} = {[]};
    for k = 1:numel(set)
        R{k + 1} = num2cell(nchoosek(set, k), 2);
    end
    S = cat(1, R{:});
end
