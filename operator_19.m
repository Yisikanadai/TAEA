function [info,pop,fit]=operator_19(pop,data,info,fit)
    [pop_size,~]=size(pop);
    for x=1:info.tl1
        [info,popout]=RL_GWO2(pop,info,fit);
        fitnew=decode_01(popout,info,data);
        for i=1:pop_size
            if fitnew(i)<fit(i)
                pop(i,:)=popout(i,:);
                fit(i)=fitnew(i);
            end
        end
    end
end
    