function [info, sol, sch] = heft_sol(info, data)
    info = setuppara_RL(info);
    info.nt = 100;
    info.pc = 0.9;
    info.pm = 0.2;
    info.n = data.n;
    info.m = data.m;
    info.cmax = data.cmax;

    dataheft = data;
    info.m = dataheft.m;
    info.n = size(dataheft.ct, 1);
    info.cmax = data.cmax;
    dataheft.aft = dataheft.hx;
    dataheft.w = dataheft.ct;
    dataheft.ckef = dataheft.c;
    dataheft.c = dataheft.st;
    dataheft.pre = dataheft.xh;
    dataheft.pkind = dataheft.p;
    dataheft.ps = dataheft.pkind;

    info.rank = rankqueen1(dataheft, info);
    sch = heft(dataheft, info);
    sol = initial_sol(sch, data);
    sol = loc_energy(sol, data, info);
end