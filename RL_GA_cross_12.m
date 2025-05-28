function [info,popout]=RL_GA_cross_12(pop,info,fit)
    [ps,len]=size(pop);
    ncr=round(ps*info.pc);
    popout=pop;
    f=1./fit;
    f=f./sum(f);
    f=cumsum(f);
    for i=1:1:ncr*2
        i1=find(rand<=f,1,'first');
        i2=find(rand<=f,1,'first');
        j=randi([1 len]);
        popout(i1,:)=[pop(i1,1:j) pop(i2,j+1:end)];
        popout(i2,:)=[pop(i2,1:j) pop(i1,j+1:end)];
    end
end
    