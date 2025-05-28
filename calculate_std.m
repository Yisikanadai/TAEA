function [standardDeviation, mean_fit, fit] = calculate_std(pop, info, data)
    fit = decode_01(pop, info, data);
    stdfit = std(fit);
    standardDeviation = stdfit;
    mean_fit = mean(fit);
end