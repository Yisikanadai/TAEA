function [info, popout] = RL_GA_mutation2_13(pop, info)
    [ps, len] = size(pop);
    nmu = round(ps * info.pm);
    popout = pop;
    for i = 1:1:nmu * 3
        j = randi([1 ps]);
        j1 = randi([1 len - 1]);
        j2 = randi([j1 + 1 len]);
        popout(j, j1) = rand(1, 1);
        popout(j, j2) = rand(1, 1);
    end
end