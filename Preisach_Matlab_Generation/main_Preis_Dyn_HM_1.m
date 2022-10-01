%Dinamikus Preisach modell, derivalt analitikusan kozelitve
clear all;

mu_0 = 4*pi*1e-7;

% --------------------------------------------------------------------
HB  = load('HB_frec_20TO500.txt');
par = load('param_mh.txt');

n_frec = 8;

frec     = [20 50 100 150 200 300 400 500]; %Hz
ind_H    = [1 3 5 7 9 11 13 15];

n_i = par(1); %the number of Preisach functions 
par = par(2:length(par));

% a dinamikus preisach parameterei
par_ambmcm = load('Param_HB_Dyn_4.txt');
am = par_ambmcm(1);
bm = par_ambmcm(2);
cm = par_ambmcm(3);

%for plotting
Hmin = -100;
Hmax = 400;
Bmin = 0;
Bmax = 2;

col = [1 0 0
       0 1 0
       0 0 1
       1 0.8 0.2
       0.6 0.2 0
       0.1 1 1
       0 0 0
       1 0.6 0];

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

n_periods = 10;
n_HB   = size(HB);
n_HB_1 = n_HB(1) - 1;
n_H    = n_HB(1)*n_periods - n_periods;

%the ramp of H
n_ramp_per = 5;

%stracaise line
n_m  = 2;

H      = zeros(n_frec,n_H);
B      = zeros(n_frec,n_H);
M      = zeros(n_frec,n_H);
Hm     = zeros(n_frec,n_H);
dMpdHm = zeros(n_frec,n_H);
dt     = zeros(1,n_frec);

F_cost = 0;
for ind_frec = 1:n_frec
    T  = 1/frec(ind_frec);
    dt(ind_frec) = T/n_HB(1);

    for i = 1:n_periods
        H(ind_frec,(i-1)*n_HB_1+1:i*n_HB_1) = HB(2:n_HB(1),ind_H(ind_frec));
    end

    ramp = linspace(0,1,n_ramp_per*n_HB(1));
    ramp = [ramp ones(1,(n_periods - n_ramp_per)*n_HB(1) - n_periods)];
    
    H(ind_frec,:) = H(ind_frec,:).*ramp;

    Lj   = zeros(n_m,2); %a lepcsosgorbe inicializalasa
    Lj_r = Lj;

    H_r    = 0;
    Hm_r   = 0;
    dMpdHm(ind_frec,1) = kimenet_dMpdH_AnEv_abc(par_abc,n_i,Lj);
    dMpdHm_r  = dMpdHm(ind_frec,1);

    amORdtDIV2   = am*dt(ind_frec)*0.5;
    mu_0ORbm     = mu_0*bm;
    mu_0ORbmDIV2 = mu_0*bm*0.5;
    for i = 1:n_H
        hiba = 1;
      
        while (hiba > 1e-4)
              %the modified magnetic field intensity, which is the input of the
              %hysteresis model
              num_1 = 1 + mu_0ORbmDIV2*(dMpdHm(ind_frec,i) + dMpdHm_r);
              Hm(ind_frec,i) = ( (num_1 - amORdtDIV2)*Hm_r + (amORdtDIV2 + cm - mu_0ORbm)*H(ind_frec,i) + (amORdtDIV2 - cm + mu_0ORbm)*H_r )/(num_1 + amORdtDIV2);

              %set the starcaise line for Hm
              [Lj,nLa] = vectf_all_1(Lj_r,Hm(ind_frec,i));

              %the derivative in function of Hm
              dMpdHm_1 = kimenet_dMpdH_AnEv_abc(par_abc,n_i,Lj);

              %calculate the difference
              delta     = dMpdHm_1 - dMpdHm(ind_frec,i);
              %update dMpdHm
              dMpdHm(ind_frec,i) = dMpdHm(ind_frec,i) + 0.5*delta; 

              %error
              if delta > 0
                  hiba = delta;
              else
                  hiba = -delta;
              end

              %str = sprintf('%d  i = %d, hiba = %g, diff = %g',ind_frec,i,hiba,dMpdHm_1 - dMpdHm(ind_frec,i));
              %disp(str);
        end
        dMpdHm(ind_frec,i) = dMpdHm_1;%kimenet_dMpdH_AnEv(par,n_i,Lj);
        dMpdHm_r  = dMpdHm(ind_frec,i); 
          
        %calculate the magnetization corresponding to Hm
        M(ind_frec,i) = kimenet_ClForm_abc(par_abc,n_i,Lj);
        B(ind_frec,i) = mu_0*(Hm(ind_frec,i) + M(ind_frec,i));
      
        Hm_r = Hm(ind_frec,i);
        H_r  = H(ind_frec,i);
        Lj_r = Lj;
    end
    
    F_cost = F_cost + sum(( B(ind_frec,n_HB(1)*(n_periods-1) - (n_periods-1):n_H) - HB(:,ind_H(ind_frec)+1)').^2)/n_H;
    
    fprintf('ind_frec = %d\n',ind_frec);
end
fprintf('F_cost = %e\n',F_cost);
        
%plot the hysteresis loop
figure 
    set(gcf,'Color',[1,1,1]);
    hold on;
    for ind_frec = 1:n_frec
        plot(HB(:,ind_H(ind_frec)),HB(:,ind_H(ind_frec)+1),'Color','b','LineWidth',2);
        plot(H(ind_frec,n_HB(1)*(n_periods-1) - (n_periods-1):n_H),B(ind_frec,n_HB(1)*(n_periods-1) - (n_periods-1):n_H),'Color','r','LineWidth',2,'LineStyle','-');
    end
    grid on;
    box on;
    set(gca,'FontSize',24);
    axis([Hmin Hmax Bmin Bmax]);
    set(gca,'XTick',linspace(Hmin,Hmax,5));
    set(gca,'YTick',linspace(Bmin,Bmax,5));
    xlabel('\it H \rm(A/m)','fontname','Times New Roman','fontsize',32);
    ylabel('\it B \rm(T)','fontname','Times New Roman','fontsize',32);
    