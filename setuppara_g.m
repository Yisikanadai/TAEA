function info=setuppara_g(info,data)
info.maxrt = 120;
if info.num>=14
    info.maxrt=720;
elseif info.num>=13
    info.maxrt=480;
elseif info.num>=11
    info.maxrt=420;
elseif info.num>=5
    info.maxrt=240;
end
info.np = 40;
info.pc = 0.9;
info.pm = 0.2;
info.tf = 500;
info.tg = 100;
info.tl = 10;
info.ng = 4000;
info.n = data.n;
info.m = data.m;
info.cmax = data.cmax;
if info.num <= 4
    info.tmax = 1000;
elseif info.num <= 8
    info.tmax = 2000;
elseif info.num <= 10
    info.tmax = 3000;
elseif info.num <= 16
    info.tmax = 8000;
end
end