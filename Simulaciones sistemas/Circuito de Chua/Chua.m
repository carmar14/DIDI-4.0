function dXdt=Chua(t,X)  %  
    %Parametros del sistema
    a=9.5; %9.5
%     B=9; %sonidos mas agudos B pequeños, B grandes (antes del caos) sonidos agudos
%     a=10.91865; %caos;
    B=14; %17
%     a=6;
%     B=8;
    
    

    d1=1/6;
    d3=1/16;
    %x1=X(1), x2=X(2), x3=X(3)
    phi=-d1*X(1)+d3*X(1)^3;
    dXdt=[a*(X(2)-phi); X(1)-X(2)+X(3); -B*X(2)];

end