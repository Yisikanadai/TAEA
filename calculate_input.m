function [inputFeatures, fit] = calculate_input(pop, info, data, fit, fitold)
    if info.mode == 1
        min_fit = [129.5 182.5 203.9 208.4 512.4 357.2 485.6 694.8 ...
                767.32 746.49 2000 2000 2000 1500 2000 1500];
    elseif info.mode == 2
        min_fit = [166.6 182.7 552.7 304.9 330.1 384.3 646.6 753.8 ...
                819.15 843.23 1500 1500 2000 1000 2500 2000];
    end
    standardDeviation = std(fit) / 1000;
    mean_fit = min_fit(info.num) / mean(fit);
    min_fit = min_fit(info.num) / min(fit);
    sum_ev = 0;
    for i = 1:info.np
        sum_ev = sum_ev + (fitold(i) - fit(i));
    end
    E_V = sqrt(info.np * (min(fitold) - min(fit))) / (sum_ev + 1);
    inputFeatures = [standardDeviation, mean_fit, min_fit, E_V];
end