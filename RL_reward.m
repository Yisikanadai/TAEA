function reward = RL_reward(fit, fitold, info, stepcount)
bestOld = min(fitold);
bestNew = min(fit);
switch info.mode
    case 1
        prior = [129.5 182.5 203.9 208.4 512.4 357.2 485.6 694.8 ...
                 767.32 746.49 2000 2000 2000 1500 2000 1500];
    case 2
        prior = [166.6 182.7 552.7 304.9 330.1 384.3 646.6 753.8 ...
                 819.15 843.23 1500 1500 2000 1000 2500 2000];
    otherwise
        error('Unsupported info.mode');
end
f_opt = prior(info.num);
lambd = 0.001;
k0 = 0.002;
c = 0.5;
ke = k0/(1+lambd*(stepcount-1));
dif = bestOld-bestNew;
base = -(f_opt-bestNew);
reward = ke*base*(1-exp(-dif/c));
end