function W = initializeHe(sz, fan_in)
stddev = sqrt(2 / fan_in);
W = randn(sz) * stddev;
end