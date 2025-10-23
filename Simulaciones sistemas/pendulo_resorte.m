clc
clear
close all

%Par�metros
global m l k g
m=10;
l=0.5;
k=1000;
g=9.8;

%%X(1)=r X(2)=rp X(3)=theta X(4)=thetap
%x0=[-0.1, 0, pi/2, 0];
% x0=[-0.1, 0, pi/8, 0];
x0=[0.7, 0, pi/6, 0]; 
%x0=[0.5, 3, -pi/3, -4];
t=0:4e-3:10;

[t,y]=ode45(@pendulo_resorte_nl,t,x0);

[m1 n]=size(y);
%animacion
figure
r=y(:,1);
theta=y(:,3);
xpos=r.*sin(theta);
ypos=-r.*cos(theta);
for i=1:m1
    x_b=[0 xpos(i)];
    y_b=[0 ypos(i)];
    plot(x_b,y_b,'b')
    hold on
    plot(xpos(1:i),ypos(1:i),'.b')
    plot(x_b(2),y_b(2),'.b','MarkerSize',30)
    hold off
%     xlim([min(xpos) max(xpos)])
%     ylim([min(ypos) 0])
    xlim([min(xpos)-0.1 0.1+max(xpos)])
    if max(ypos)<0
        ylim([min(ypos) 0])
    else
        ylim([min(ypos) max(ypos)])
    end
    pause(0.001)
end


figure
for i=1:n
    subplot(n,1,i)    
    plot(t,y(:,i))
    if i==1
        ylabel('$r (m)$','interpreter','latex')
        xlabel('t(seg)')
    elseif i==2
        ylabel('$\dot{r} (m/s)$','interpreter','latex')
        xlabel('t(seg)')
     elseif i==3
        ylabel('$\theta  (rad)$','interpreter','latex') 	
        xlabel('t(seg)')
    else
        ylabel('$\dot{\theta} (rad/s)$','interpreter','latex')
        xlabel('t(seg)')
    end
        
end


function [dxdt]=pendulo_resorte_nl(t,X)
    global m l k g
    
    %X(1)=r X(2)=rp X(3)=theta X(4)=thetap
    x1p=X(2);
    x3p=X(4);
    x2p=g*cos(X(3))-(k/m)*(X(1)-l)+ X(1)*X(4)^2;
    x4p=-(2*X(2)*X(4)+g*sin(X(3)))/X(1);
    dxdt=[x1p;x2p;x3p;x4p];
end
