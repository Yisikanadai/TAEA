function reward=RL_reward(fit,fitold,info,stepcount)
    bestOld=min(fitold);
    bestNew=min(fit);
    switch info.mode
        case 1
            prior=[181.51 195.26 220.51 269.26 300 325.92 372.45 402.79 ...
                767.32 746.49 2000 2000 2000 2000 2200 2200];
        case 2
            prior=[198.48 142.96 343.69 266.47 382.24 274.59 791.44 686.03 ...
                919.15 743.23 2000 1700 2400 3000 3000 3000];
        otherwise
            error('Unsupported info.mode');
    end
    f_opt=prior(info.num);
    lambd=0.001;
    k0=0.002;
    ke=k0/(1+lambd*(stepcount-1));
    dif=bestOld-bestNew;
    base=-(f_opt-bestNew);
    c=0.5;
    reward=ke*base*(1-exp(-dif/c));
end
    