% Stepwise selection method
function selected_ind = stepwise_selection(A, Y)
    % number of features
    n = size(A, 2);
    
    % initially no covariate is selected
    remaining_ind = [1:n];
    selected_ind = [];

    % First perform 2 steps of forward selection
    index = forward_selection(A, Y, selected_ind, remaining_ind);
    % if nothing selected
    if (index == 0)
        return
    end
    % otherwise
    selected_ind = [index];
    remaining_ind(remaining_ind == index) = [];

    % One more step of forward selection
    index = forward_selection(A, Y, selected_ind, remaining_ind);
    % if nothing selected
    if (index == 0)
        return
    end
    % otherwise
    selected_ind = [selected_ind, index];
    remaining_ind(remaining_ind == index) = [];

    while true
        % Perform one step of backward elimination
        index = backward_elimination(A, Y, selected_ind, remaining_ind);
        % if something is to be eliminated
        if (index > 0) 
            selected_ind(selected_ind == index) = [];
            remaining_ind = [remaining_ind, index];
        end

        % Perform one step of forward selection
        index = forward_selection(A, Y, selected_ind, remaining_ind);
        % if nothing selected
        if (index == 0)
            return
        end
        % otherwise
        selected_ind = [selected_ind, index];
        remaining_ind(remaining_ind == index) = [];
    end
end