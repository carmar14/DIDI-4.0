clc
close all
clear

global g l a w
g=9.8;
l=0.8;
a= 0.4;
w=1; %w=0 pendulo simple 2.9 caos

t=0:0.01:20;

%estados: x1=theta, x2=theta_p
X0=[pi/4; 0];
% X0=[2.0944; 0];
[t y]=ode45(@pend_mov_cir,t,X0);

[m n]=size(y);

for i=1:n
    subplot(n,1,i)
    plot(t,y(:,i))
end

pause(1)
figure
x_p=a*cos(w*t);
y_p=a*sin(w*t);

x_m=a*cos(w*t)+l*sin(y(:,1));
y_m=a*sin(w*t)-l*cos(y(:,1));
xinf=-a+min(x_m);
xsup=a+max(x_m);
yinf=-a+min(y_m);
ysup=a+max(y_m)+a*0.8;

for i=1:m
    
    x1=a*cos(w*t(i));
    y1=a*sin(w*t(i));
    
    xm=a*cos(w*t(i))+l*sin(y(i,1));
    ym=a*sin(w*t(i))-l*cos(y(i,1));
    
    x_c=[x1 xm];
    y_c=[y1 ym];
    
    %----observar la trayectoria de la bola
    X(i)=xm;
    Y(i)=ym;
    
    plot(x_c,y_c,'r')
    hold on
    plot(xm,ym,'.b', 'MarkerSize',30)
    plot(x_p,y_p,'k','MarkerSize',10)
    
    xlim([xinf xsup]);
    ylim([yinf ysup]);
    plot(X(1:i),Y(1:i),'.b')
    hold off    
    pause(0.0001)
end


function xp =pend_mov_cir(t,x)
global g l a w
x1=x(1);
x2=x(2);

x1p=x2;
x2p=-(g/l)*sin(x1)+(a*w^2/l)*cos(x1-w*t);

xp=[x1p;x2p];
end