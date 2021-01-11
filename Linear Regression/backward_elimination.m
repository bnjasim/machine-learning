function index = backward_elimination(A, Y, selected_ind, remaining_ind)
    % number of data points
    m = size(A, 1);
    % number of features
    n = size(A, 2);
    F_to_remove = 0.05;
    index = 0;

    % Compute partial F scores by removing each of the selected indices
    partial_F_scores = zeros(size(selected_ind));

    for i = 1:length(selected_ind)
        s_ind = selected_ind(i);
        tmp_selected_ind = selected_ind(selected_ind ~= s_ind);

        A1 = A(:, tmp_selected_ind);
        A2 = A(:, selected_ind);

        deg_freedom2 = m - length(selected_ind) - 1;

        rss1 = RSS(A1, Y);
        rss2 = RSS(A2, Y);

        partial_F_scores(i) = (rss1 - rss2)/rss2 * deg_freedom2;

    end

    % Which is the largest among partial_F_scores ?
    [max_F, ind_F] = max(partial_F_scores);

    if (max_F <= F_to_remove)
        index = selected_ind(ind_F);
        fprintf(1, "Backward %d\n", index);
    end
end