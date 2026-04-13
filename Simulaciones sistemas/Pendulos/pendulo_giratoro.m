clc
close all
clear

global g l a w
g=9.8;
l=10;%0.8;
w=1.2; % 1.5 caos

t=0:0.01:20;

%estados: x1=theta, x2=theta_p
theta=deg2rad(15);
X0=[theta; 0];
% X0=[2.0944; 0];
[t y]=ode45(@pend_gir,t,X0);

[m n]=size(y);

for i=1:n
    subplot(n,1,i)
    plot(t,y(:,i))
end

pause(1)
figure
x_p=0;
y_p=0;
z_p=0;

x_m=l*sin(y(:,1)).*cos(w*t);
y_m=l*sin(y(:,1)).*sin(w*t);
z_m=-l*cos(y(:,1));

xinf=min(x_m);
xsup=max(x_m);
yinf=min(y_m);
ysup=max(y_m);
zinf=min(z_m);
zsup=max(z_m);

for i=1:m
    
    xm=l*sin(y(i,1))*cos(w*t(i));
    ym=l*sin(y(i,1))*sin(w*t(i));
    zm=-l*cos(y(i,1));
    
    x_c=[x_p xm];
    y_c=[y_p ym];
    z_c=[z_p zm];
    
    %----observar la trayectoria de la bola
    X(i)=xm;
    Y(i)=ym;
    Z(i)=zm;
    
    plot3(x_c,y_c,z_c,'r')
    hold on
    plot3(xm,ym,zm,'.b', 'MarkerSize',30)
    plot3(x_p,y_p,z_p,'k','MarkerSize',10)
    
    xlim([xinf xsup]);
    ylim([yinf ysup]);
    if zsup < 0
        zlim([zinf 0])
    else
        zlim([zinf zsup]);
    end
    
    
    plot3(X(1:i),Y(1:i),Z(1:i),'.b')
    xlabel('x');
    ylabel('y');
    zlabel('z');
    grid on
    view(60,25)
    hold off
    pause(0.0001)
end


function xp =pend_gir(t,x)
global g l w
x1=x(1);
x2=x(2);

x1p=x2;
x2p=-(g/l)*sin(x1)+w^2*cos(x1)*sin(x1);

xp=[x1p;x2p];
end