function dMpdH = kimenet_dMpdH_AnEv_abc(par,n,Lj)

nLj = size(Lj);
H   = Lj(nLj(1),1);

dMpdH = 0;
if nLj(1) > 1
    j = 1;
    for i = 1:n
        a = par(j);
        b = par(j+1);
        c = par(j+2);

        if abs( Lj(nLj(1),2) - Lj(nLj(1)-1,2) )<1e-9 %a Lepcsosgorbe utolso szakasza vizszintes, felfele haladok
           H_r = Lj(nLj(1)-1,1);

           par_1 = exp(-b*H);

           dMpdH = dMpdH + 2*a^2/(b*c)*par_1/(1 + c*par_1)^2*( 1/(1 + c*exp(b*H_r)) - 1/(1 + c*exp(b*H)) );
        else    % a lepcsosgorbe utolso szakasza fuggoleges, lefele haladok
           H_r = Lj(nLj(1)-1,2);

           par_1 = exp(b*H);

           dMpdH = dMpdH + 2*a^2/(b*c)*par_1/(1 + c*par_1)^2*( 1/(1 + c*exp(-b*H_r)) - 1/(1 + c*exp(-b*H)));
        end 

        j = j + 3; 
    end
end

n_d     = length(par);
n_x_rev = (n_d - 1 - 3*n)/2;
dMpdH_rev   = abs(par(n_d)); % (d*H)' = d
j = 3*n + 1;
for i = 1:n_x_rev
     OneDIVK2  = 1/abs(par(j+1));
     dMpdH_rev = dMpdH_rev + abs(par(j))*OneDIVK2*( 1 - tanh(H*OneDIVK2).^2 );
     j = j + 2;
end
dMpdH = dMpdH + dMpdH_rev;