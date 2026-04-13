clc
clear
close all

m1=0.5;
m2=0.5;
l1=0.5; %1;
l2=0.5; %0.6;
g=9.8;

%condiciones iniciales
t1ini=deg2rad(100);
t1pini=deg2rad(0);
t2ini=deg2rad(-110); %(16)
t2pini=deg2rad(0);

sim('pend_dob.slx');

t1=ans.theta.signals.values(:,1);
t2=ans.theta.signals.values(:,2);
t=ans.tout;

% plot(t,t1)

[m n]=size(t1);

figure
x1=0;
y1=0;
x_m1=l1*sin(t1);
y_m1=-l1*cos(t1);
x_m2=l1*sin(t1)+l2*sin(t2);
y_m2=-l1*cos(t1)-l2*cos(t2);

% xinf=min(x_m1);
% xsup=max(x_m1);
% yinf=min(y_m1);
% ysup=max(y_m1);

% xinf=min(x_m2);
% xsup=max(x_m2);
% yinf=min(y_m2);
% ysup=max(y_m2);

xinf=-(l1+l2);
xsup=(l1+l2);
yinf=-(l1+l2);
ysup=(l1+l2);

for i=1:m
    
    
    
    xm1=l1*sin(t1(i));
    ym1=-l1*cos(t1(i));
    
    xm2=l1*sin(t1(i))+l2*sin(t2(i));
    ym2=-l1*cos(t1(i))-l2*cos(t2(i));
    
    x_c1=[x1 xm1];
    y_c1=[y1 ym1];
    
    x_c2=[xm1 xm2];
    y_c2=[ym1 ym2];
    
    %----observar la trayectoria de la bola
    X1(i)=xm1;
    Y1(i)=ym1;
    
    X2(i)=xm2;
    Y2(i)=ym2;
    
    plot(x_c1,y_c1,'r')
    hold on
    plot(xm1,ym1,'.r', 'MarkerSize',30)
    plot(x_c2,y_c2,'k')
    plot(xm2,ym2,'.k', 'MarkerSize',30)
    
%     xlim([xinf xsup]);
%     if ysup < 0
%         ylim([yinf 0])
%     else
%         ylim([yinf ysup]);
%     end
    plot(X1(1:i),Y1(1:i),'.r')
    plot(X2(1:i),Y2(1:i),'.k')
    hold off
    pause(0.0001)
end
