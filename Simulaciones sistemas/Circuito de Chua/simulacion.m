clc
clear
close all
Xini=[0.1 0.1 0.2];
Fs=44100;
t=0:0.01:500;
[t,y]=ode45(@Chua,t,Xini);
N=length(t);
subplot(4,1,1)
plot(t,y(:,1))
title('Tension C_1')
ylabel("v_{C1}(V)");
xlabel('Tiempo (s)');
% xlim([0 100])
subplot(4,1,2)
plot(t,y(:,2))
title('Tension C_2')
ylabel("v_{C2}(V)");
xlabel('Tiempo (s)');
subplot(4,1,3)
plot(t,y(:,3))
title('Corriente en L')
ylabel("i_L(A)");
xlabel('Tiempo (s)');

%Generando audio con voltaje en el capacitor 1
%normalizar se�al para audio usando la ecuacion de una linea recta y=mx+b
%para no saturar
m=2/(max(y(:,1))-min(y(:,1)));
b=1-m*max(y(:,1));
yn=m*y(:,1)+b;

%suma de vc's
% m=2/(max(y(:,1)+y(:,2))-min(y(:,1)+y(:,2)));
% b=1-m*max(y(:,1)+y(:,2));
% yn=m*(y(:,1)+y(:,2))+b;

% audiowrite('audio1.wav',y(:,1),Fs);
audiowrite('audio1.wav',yn,Fs);
[ya,Fs] = audioread('audio1.wav');
sound(ya,Fs);
subplot(4,1,4)
plot(t,ya)
title('Audio')
