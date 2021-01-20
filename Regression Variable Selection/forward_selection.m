function index = forward_selection(A, Y, selected_ind, remaining_ind)
    % number of data points
    m = size(A, 1);
    % number of features
    n = size(A, 2);
    F_to_enter = 0.05;
    index = 0;

    % Compute partial F scores by adding each of the remaining indices 
    partial_F_scores = zeros(size(remaining_ind));

    for i = 1:length(remaining_ind)
        r_ind = remaining_ind(i);
        tmp_selected_ind = [selected_ind, r_ind];

        A1 = A(:, selected_ind);
        A2 = A(:, tmp_selected_ind);

        deg_freedom2 = m - length(tmp_selected_ind) - 1;

        rss1 = RSS(A1, Y);
        rss2 = RSS(A2, Y);

        partial_F_scores(i) = (rss1 - rss2)/rss2 * deg_freedom2;

    end

    % Which is the largest among partial_F_scores ?
    [max_F, ind_F] = max(partial_F_scores);

    if (max_F > F_to_enter)
        index = remaining_ind(ind_F);
        fprintf(1, "Forward %d\n", index);
    end
end