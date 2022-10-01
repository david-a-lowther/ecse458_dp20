%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Preisach model with closed form Everett
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function M = kimenet_ClForm_abc(par,n,Lj)

n_Lj = size(Lj);
M    = 0;
if n_Lj(1)>1
   
   %ind_Lj = 1:n_Lj(1)-1; 
   %x = Lj(ind_Lj,1);
   %y = Lj(ind_Lj,2);
   
   %this seems to be faster, but one point of Ev (h1 == h2) is computed in excess
   x = Lj(:,1);
   y = Lj(:,2);
   
   Ev = 0;
   j = 1;
   for i = 1:n
        a = par(j);
        b = par(j+1);
        c = par(j+2);
        
        par_1 = exp(b*x);
        par_2 = exp(b*y);
        
        if c == 1
            Ev = Ev + 0.5*(a/b)^2*( (par_1 - par_2).^2 )./( ((1.0 + par_1).^2) .* ((1.0 + par_2).^2) );
        else
            cORcMin1 = c*c - 1.0;
            par_3 = (1.0 + c*par_1).*(c + par_2);
            
            Ev = Ev - (a/b)^2*( cORcMin1*(par_1 - par_2) + par_3.*log( (1.0 + c*par_2).*(c + par_1)./par_3) )./...
                      (cORcMin1^2*par_3);
        end
        j = j + 3;     
   end
   
   if Lj(1,1) == Lj(2,1)
         M = -Ev(1);
   else
         M = Ev(1);
   end
   
   for i = 2:n_Lj(1)-1
        if Lj(i,1) == Lj(i+1,1)
           %M = M - 2*Ev(i);
           M = M - (Ev(i) + Ev(i)); %seems to be faster
        elseif Lj(i,2) == Lj(i+1,2)
           %M = M + 2*Ev(i);    
           M = M + (Ev(i) + Ev(i)); %seems to be faster
        end
   end
end

n_d     = length(par);
n_x_rev = (n_d - 1 - 3*n)/2;
M_rev   = abs(par(n_d))*Lj(n_Lj(1),1); %d*H
j = 3*n + 1;
for i = 1:n_x_rev
     M_rev = M_rev + abs(par(j))*tanh(Lj(n_Lj(1),1)/abs(par(j+1)));
     j = j + 2;
end
M = M + M_rev;
