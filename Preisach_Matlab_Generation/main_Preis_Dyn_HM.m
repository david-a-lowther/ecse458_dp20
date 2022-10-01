%Dinamikus Preisach modell, Lornent eloszlassal, derivalt analitikusan kozelitve
clear all;

mu_0 = 4*pi*1e-7;

% --------------------------------------------------------------------
HB  = load('HB_frec_20TO500.txt');
par = load('param_mh.txt');

ind_frec = 5;

frec     = [20 50 100 150 200 300 400 500]; %Hz
ind_H    = [1 3 5 7 9 11 13 15];

n_i = par(1); %the number of Preisach functions 
par = par(2:length(par));

%the parameters of the dynamic mnodel
am = 3056.5;
bm = 99.29;
cm = 0.69;

%for plotting
Hmin = -300;
Hmax = 300;
Bmin = -2;
Bmax = 2;

% --------------------------------------------------------------------
%convert the parameters
par_abc = par;
j = 1;
for i = 1:n_i
    par_abc(j+1) = 1.0/par(j+2);
    par_abc(j+2) = exp(par(j+1)*par_abc(j+1));
    par_abc(j)   = par(j)*par_abc(j+2);
    j = j+3;
end

Hsat = max(HB(:,ind_H(ind_frec)));  %the saturation - (limit of the Preisach triangle - technical saturation for convenience)

n_periods = 12;
n_HB   = size(HB);
n_HB_1 = n_HB(1) - 1;
n_H    = n_HB(1)*n_periods - n_periods;
H      = zeros(n_H,1);

T  = 1/frec(ind_frec);
dt = T/n_HB(1);

for i = 1:n_periods
    H((i-1)*n_HB_1+1:i*n_HB_1) = HB(2:n_HB(1),ind_H(ind_frec));
end

n_ramp_per = n_periods - 2;
ramp = linspace(0,1,n_ramp_per*n_HB(1));
ramp = [ramp ones(1,(n_periods - n_ramp_per)*n_HB(1) - n_periods)];
    
H = H.*ramp';

n_m  = 2;
nLa  = 1;
Lj   = zeros(n_m,2); %a lepcsosgorbe inicializalasa
Lj_r = Lj;

B      = zeros(1,n_H);
M      = zeros(1,n_H);
Hm     = zeros(1,n_H);
dMpdHm = zeros(1,n_H);
H_r    = 0;
Hm_r   = 0;
dMpdHm(1) = kimenet_dMpdH_AnEv_abc(par_abc,n_i,Lj);
dMpdHm_r  = dMpdHm(1);

amORdtDIV2   = am*dt*0.5;
mu_0ORbm     = mu_0*bm;
mu_0ORbmDIV2 = mu_0*bm*0.5;
for i = 1:n_H
      hiba = 1;
      
      while (hiba > 1e-6)
          %the modified magnetic field intensity, which is the input of the
          %hysteresis model
          num_1 = 1 + mu_0ORbmDIV2*(dMpdHm(i) + dMpdHm_r);
          Hm(i) = ( (num_1 - amORdtDIV2)*Hm_r + (amORdtDIV2 + cm - mu_0ORbm)*H(i) + (amORdtDIV2 - cm + mu_0ORbm)*H_r )/(num_1 + amORdtDIV2);

          %set the starcaise line for Hm
          [Lj,nLa] = vectf_all_1(Lj_r,Hm(i));

          %the derivative in function of Hm
          dMpdHm_1 = kimenet_dMpdH_AnEv_abc(par_abc,n_i,Lj);
          
          %calculate the difference
          delta     = dMpdHm_1 - dMpdHm(i);
          %update dMpdHm
          dMpdHm(i) = dMpdHm(i) + 0.5*delta; 
          
          %error
          if delta > 0
              hiba = delta;
          else
              hiba = -delta;
          end
          
          %str = sprintf('i = %d, hiba = %g, diff = %g',i,hiba,dMpdHm_1 - dMpdHm(i));
          %disp(str);
      end
      dMpdHm(i) = dMpdHm_1;%kimenet_dMpdH_AnEv(par,n_i,Lj);
      dMpdHm_r  = dMpdHm(i); 
          
      %calculate the magnetization corresponding to Hm
      M(i)     = kimenet_ClForm_abc(par_abc,n_i,Lj);
      B(i)     = mu_0*(Hm(i) + M(i));
      
      Hm_r = Hm(i);
      H_r  = H(i);
      Lj_r = Lj;
      
      str = sprintf('i = %d',i);
      disp(str);
end
 
% Create CSV files from output (H, B, HB):
writematrix(H, 'H.csv')
for k1 = 1:length(H)
    fprintf('Number at position in H %d = %6.2f\n', k1, H(k1));
end
B = reshape(B, length(B), 1);
writematrix(B, 'B.csv');
for k2 = 1:length(B)
    fprintf('Number at position in B %d = %6.2f\n', k2, B(k2));
end
writematrix([H,B], 'HB.csv')

%plot the hysteresis loop
figure 
    set(gcf,'Color',[1,1,1]);
    h3 = plot(H,B,'Color','r','LineWidth',2);
    hold on;
    grid on;
    box on;
    set(gca,'FontSize',14);
    axis([Hmin Hmax Bmin Bmax]);
    set(gca,'XTick',linspace(Hmin,Hmax,5));
    set(gca,'YTick',linspace(Bmin,Bmax,5));
    xlabel('\it H \rm(A/m)','fontname','Times New Roman','fontsize',24);
    ylabel('\it B \rm(T)','fontname','Times New Roman','fontsize',24);
    title(['f = ', int2str(frec(ind_frec)), '  Hz' ]);