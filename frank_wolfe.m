function [optval,xopt,it] = frank_wolfe(lmi_set,opts,Rvar,x0,epsilon,verbose)
%[optval,xopt,it] = frank_wolfe(lmi_set,opts,Rvar,x0,epsilon,verbose);
% inputs: lmi_set        -> set of lmis, provided by ``getlmis''
%         opts           -> options vector, as in ``mincx''
%         Rvar           -> defines the objective fun. min(1/det(Rvar)))
%         x0        [opt]-> initial condition [from feasp]
%         epsilon   [opt]-> desired accuracy [1e-3]
%         verbose   [opt]-> verbose [1]


% outputs: optval        -> global optimal value for -log(det(Rvar))
%          xopt          -> decision variables
%          it            -> number of iterations
% Date: 04/11/2019
% Author: Lucas N. Egidio
% Email:  lucas.n.egidio@gmail.com


% No epsilon provided
    if(~exist('verbose','var')|| isempty(verbose))
        verbose = 1; 
    end
    
    % No epsilon provided
    if(~exist('epsilon','var')|| isempty(epsilon))
        epsilon = 1e-3; 
    end
    
    % No initial condition provided
    if(~exist('x0','var') || isempty(x0))
        [tmin,x0] = feasp(lmi_set,opts);
        if tmin>0
            xopt = [];
            optval = [];
            return;
        end

    end
    
    Rx  = dec2mat(lmi_set,x0,Rvar);
    outx = x0;
    optval = log(det(Rx));
    
    it = 1;
    while 1
        n_dec = decnbr(lmi_set);
        c = zeros(n_dec,1);
        for i=1:n_dec
            Rn =  defcx(lmi_set,i,Rvar);  
            c(i) = -trace(Rx\Rn );
        end
        
        [~,xopt] = mincx(lmi_set,c,opts,x0);
        out = xopt;
        if( isempty(xopt))
            keyboard
        end
        R = dec2mat(lmi_set,xopt,Rvar);
        cond = -trace((Rx)\(R-Rx)) ;
        if cond < -epsilon
            vet = @(x_) - log(det(x_*R+(1-x_)*Rx));
            
            [alphax,optval]= unimin(vet);
            Rx =   alphax*R   + (1-alphax)*Rx;
     
            outx = alphax*out + (1-alphax)*outx;
            if(verbose)
                fprintf('%4d: %f \t %f\n',it, cond, nvol(Rx) ) %show converg.
            end
        else
            if(verbose)
                fprintf('%4d: %f \t %f\n',it, cond, nvol(Rx) ) %show converg.
            end
            xopt =outx;
            break
        end
        it = it+1;
    end

end

%% unimin - unidimentional minimization 
function [x,fval]= unimin(f,deltaM,ver,interval)
if(~exist('deltaM','var') || isempty(deltaM))
    deltaM = 1e-9;
end
if(~exist('ver','var')|| isempty(ver))
    ver = 0;
end
if(~exist('interval','var')|| isempty(interval))
    interval = [0 1];
end

phi = (1+sqrt(5))/2;
x(1) = interval(1);
x(4) = interval(2);

fval(1) = f(x(1));
fval(4) = f(x(4));

x(2) = x(4) - (x(4)-x(1))/phi;
x(3) = x(1) + (x(4)-x(1))/phi;
fval(2) = f(x(2));
fval(3) = f(x(3));
k=0;
while abs(x(1)-x(4))>deltaM
    k=k+1;
    if(ver)
        fprintf('x:\t %5.3f %5.3f %5.3f %5.3f\n',x)
        fprintf('fx:\t %5.3f %5.3f %5.3f %5.3f\n\n',fval)
    end
    if(fval(2) <fval(3))
        x(4) = x(3);
        fval(4) = fval(3);
        
        x(3) = x(2);
        fval(3) = fval(2);
        
        x(2) = x(4) - (x(4)-x(1))/phi;
        fval(2) =f(x(2));
    else
        x(1) = x(2);
        fval(1) = fval(2);
        
        x(2) = x(3);
        fval(2) = fval(3);
        
        x(3) = x(1) + (x(4)-x(1))/phi;
        fval(3) = f(x(3));
    end
end

[fval,im] = min(fval);
x=  x(im);
end