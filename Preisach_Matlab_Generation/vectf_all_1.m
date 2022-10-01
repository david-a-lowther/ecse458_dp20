% --------------------------------------------------------------------
%set the starcaise line
% --------------------------------------------------------------------
function [L, nL1] = vectf_all_1(Lj, H)

eps = 1e-8;

nLj = size(Lj);  
H_r = Lj(nLj(1),1);      

if abs(H) < 1e-16   && abs(H_r) < 1e-16
    L   = [ 0 0];
    nL1 = 1;
    return;
end

if (H >= H_r) %&& abs(H - H_r) > eps
   if Lj(1,1) ~= -1 && Lj(1,2) ~= 1
       if H > Lj(1,2) || abs(H - Lj(1,2)) < eps
          Lj = [-H H
                 H H];  
       else        
          for i = nLj(1):-1:1
             if (H <= Lj(i,2)) && (H >= Lj(i+1,2))
                  break;
             end
          end
          Lj1 = Lj;
          Lj  = [Lj1(1:i,:)
                 Lj1(i,1) H
                 H H];
       end
    else    
          for i = nLj(1):-1:1
             if (H <= Lj(i,2)) && (H >= Lj(i+1,2))
                  break;
             end
          end
          Lj1 = Lj;
          Lj  = [Lj1(1:i,:)
                 Lj1(i,1) H
                 H H];
   end    
elseif (H < H_r) %&& abs(H - H_r) > eps
    if Lj(1,1) ~= -1 && Lj(1,2) ~= 1
       if H < Lj(1,1) || abs(H - Lj(1,1)) < eps
          Lj = [H -H
                H  H];  
       else
          for i = nLj(1):-1:1
             if (H >= Lj(i,1)) && (H <= Lj(i+1,1))
                  break;
             end
          end
          Lj1 = Lj;
          Lj  = [Lj1(1:i,:)
                 H Lj1(i,2)
                 H H];                  
       end      
    else
          for i = nLj(1):-1:1
             if (H >= Lj(i,1)) && (H <= Lj(i+1,1))
                  break;
             end
          end
          Lj1 = Lj;
          Lj  = [Lj1(1:i,:)
                 H Lj1(i,2)
                 H H];                  
    end
end


nLj = size(Lj);
jel = 1;
for i = 1:nLj(1)-1
  if Lj(i,1) == Lj(i+1,1) && Lj(i,2) == Lj(i+1,2)
      jel = 0;
      break;
  end; 
end;

if jel == 0
    L = [Lj(1:i-1,1) Lj(1:i-1,2)
         Lj(i+2:nLj,1) Lj(i+2:nLj,2)];
else
    L = Lj;
end;

nL1  = size(L);